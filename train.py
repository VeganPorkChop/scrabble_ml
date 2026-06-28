"""
Training loop for ScrabbleNet.

Overview
────────
We use a replay buffer — a fixed-size pool of ExperienceTuples.  Each
training iteration:

  1. Play N self-play games IN PARALLEL across CPU cores.
  2. Add experience to the replay buffer (oldest entries dropped when full).
  3. Sample a random minibatch from the buffer.
  4. Compute MSE loss between model prediction and discounted target.
  5. Backprop + Adam optimiser step.
  6. Decay epsilon (less exploration over time).
  7. Save a checkpoint every K iterations.

Why parallel self-play?
────────────────────────
The bottleneck is NOT the GPU — it's the Python move generator (GADDAG
traversal) running on CPU.  A single game takes ~10-20 seconds.  With 16
CPU cores we can play 16 games simultaneously, cutting wall-clock time by
~10x for the self-play phase.

We use a persistent ProcessPoolExecutor with an initializer that loads
the GADDAG once per worker process.  Reloading it for every game would
cost 7 seconds each time — instead it's a one-time ~7s startup cost per
worker, then near-instant lookups afterward.

The model weights (372KB) are pickled and sent to workers each iteration
so workers can do epsilon-greedy exploitation on CPU.  The main process
keeps the GPU model for training.

Replay buffer
─────────────
Why not just train on the most recent games?
  Recent games come from a correlated sequence of states.  Training on
  correlated data causes the network to forget earlier patterns
  (catastrophic forgetting).  Sampling randomly from a large buffer
  breaks the correlation — a key insight from DeepMind's DQN (2015).

Loss: MSE.  Adam optimiser with weight decay (L2 regularisation).
"""

import argparse
import collections
import multiprocessing
import os
import random
import time
import pathlib
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.nn as nn
import torch.optim as optim

from gaddag import GADDAG
from model import ScrabbleNet
from selfplay import play_game, ExperienceTuple

# ── Defaults ──────────────────────────────────────────────────────────────────

DICT_PATH      = "TWL06.txt"
CACHE_PATH     = "gaddag.pkl"
CHECKPOINT_DIR = pathlib.Path("checkpoints")

BUFFER_SIZE    = 50_000
BATCH_SIZE     = 256
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4
GAMES_PER_ITER = 16      # should be a multiple of NUM_WORKERS
TRAIN_STEPS    = 4
ITERATIONS     = 100
EPSILON_START  = 1.0
EPSILON_END    = 0.05
SAVE_EVERY     = 10
NUM_WORKERS    = max(1, min(8, (os.cpu_count() or 4) - 2))  # leave 2 cores free


# ── Worker process setup ──────────────────────────────────────────────────────
# These must be module-level (not nested) so Python's 'spawn' start method
# can pickle them on Windows.

_worker_gaddag = None  # loaded once per worker process, lives in global

def _worker_init(cache_path: str) -> None:
    """Called once when each worker process starts.  Loads the GADDAG into a
    module-level global so it's reused across all games this worker plays."""
    global _worker_gaddag
    _worker_gaddag = GADDAG.load(cache_path)


def _worker_play(args: tuple) -> list:
    """Play one game and return a list of ExperienceTuples (as plain dicts so
    they survive the pickle round-trip back to the main process)."""
    epsilon, seed, model_state_cpu = args

    model = None
    if model_state_cpu is not None:
        m = ScrabbleNet()          # CPU model inside the worker
        m.load_state_dict(model_state_cpu)
        m.eval()
        model = m

    tuples = play_game(_worker_gaddag, model=model, epsilon=epsilon, seed=seed)

    # Tensors inside ExperienceTuple survive pickle fine, but we share them
    # as plain Python lists to avoid any PyTorch multiprocessing quirks.
    return [
        (t.board_t, t.rack_t, t.move_f, t.target)
        for t in tuples
    ]


# ── Batch builder ─────────────────────────────────────────────────────────────

def build_batch(
    samples: list[ExperienceTuple],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    boards  = torch.stack([s.board_t for s in samples]).to(device)
    racks   = torch.stack([s.rack_t  for s in samples]).to(device)
    moves   = torch.stack([s.move_f  for s in samples]).to(device)
    targets = torch.tensor(
        [s.target for s in samples], dtype=torch.float32, device=device
    ).unsqueeze(1)
    return boards, racks, moves, targets


# ── Main training loop ────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"CPU workers for self-play: {args.workers}")

    # ── Load dictionary (main process only — workers load from cache) ──
    GADDAG.load_or_build(args.dict, args.cache)   # ensures cache exists
    print(f"Dictionary ready.")

    # ── Model + optimiser ─────────────────────────────────────────────
    model = ScrabbleNet().to(device)
    print(f"ScrabbleNet parameters: {model.parameter_count():,}")

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    start_iter = 0
    latest = sorted(CHECKPOINT_DIR.glob("ckpt_*.pt"))
    if latest and not args.fresh:
        ckpt = torch.load(latest[-1], map_location=device)
        model.load_state_dict(ckpt["model"])
        start_iter = ckpt.get("iteration", 0)
        print(f"Resumed from {latest[-1]}  (iteration {start_iter})")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    loss_fn   = nn.MSELoss()

    buffer: collections.deque[ExperienceTuple] = collections.deque(maxlen=args.buffer)

    def epsilon_at(it: int) -> float:
        frac = min(it / max(args.iterations - 1, 1), 1.0)
        return EPSILON_START + frac * (EPSILON_END - EPSILON_START)

    loss_history: list[float] = []

    # ── Spawn worker pool (GADDAG loads once per worker here) ─────────
    print(f"Starting {args.workers} worker processes (loading GADDAG, ~7s each)...")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(args.cache,),
    ) as pool:
        print("Workers ready.\n")

        for it in range(start_iter, start_iter + args.iterations):
            eps = epsilon_at(it - start_iter)
            t0  = time.time()

            # 1. Self-play — all games run in parallel across workers
            #    Pass CPU copy of model weights so workers can do exploitation
            model_cpu_state = {k: v.cpu() for k, v in model.state_dict().items()} \
                              if eps < 0.99 else None

            seeds = [it * 1000 + g for g in range(args.games_per_iter)]
            task_args = [(eps, s, model_cpu_state) for s in seeds]

            new_exp = 0
            for raw_tuples in pool.map(_worker_play, task_args):
                for board_t, rack_t, move_f, target in raw_tuples:
                    buffer.append(ExperienceTuple(board_t, rack_t, move_f, target))
                    new_exp += 1

            t_selfplay = time.time() - t0

            # 2. Train on GPU
            iter_losses: list[float] = []
            if len(buffer) >= args.batch_size:
                model.train()
                for _ in range(args.train_steps):
                    samples = random.sample(list(buffer), args.batch_size)
                    boards, racks, moves, targets = build_batch(samples, device)

                    preds = model(boards, racks, moves)
                    loss  = loss_fn(preds, targets)

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    iter_losses.append(loss.item())

            avg_loss = sum(iter_losses) / len(iter_losses) if iter_losses else float("nan")
            loss_history.append(avg_loss)
            elapsed = time.time() - t0

            print(
                f"[iter {it+1:04d}]  eps={eps:.3f}  "
                f"buf={len(buffer):>6,}  +exp={new_exp:>4}  "
                f"loss={avg_loss:.4f}  "
                f"(selfplay {t_selfplay:.1f}s  total {elapsed:.1f}s)"
            )

            # 3. Save checkpoint
            if (it + 1) % args.save_every == 0:
                ckpt_path = CHECKPOINT_DIR / f"ckpt_{it+1:04d}.pt"
                torch.save({"model": model.state_dict(), "iteration": it + 1}, ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path}")

    print("\nTraining complete.")
    final_path = CHECKPOINT_DIR / "model_final.pt"
    torch.save({"model": model.state_dict(), "iteration": start_iter + args.iterations}, final_path)
    print(f"Final model saved to {final_path}")

    valid = [l for l in loss_history if l == l]
    if len(valid) >= 2:
        n = len(valid)
        split = max(1, n // 4)
        first_q, last_q = valid[:split], valid[-split:]
        print(f"\nLoss summary ({n} iters with data):")
        print(f"  First quarter avg : {sum(first_q)/len(first_q):.4f}")
        print(f"  Last quarter avg  : {sum(last_q)/len(last_q):.4f}")
    elif len(valid) == 1:
        print(f"\nLoss (1 training iter): {valid[0]:.4f}")
    else:
        print("\nNo training — buffer never reached batch size.")
        print(f"  Try --batch-size {min(64, BATCH_SIZE)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train ScrabbleNet via self-play.")
    parser.add_argument("--dict",           default=DICT_PATH)
    parser.add_argument("--cache",          default=CACHE_PATH)
    parser.add_argument("--iterations",     type=int,   default=ITERATIONS)
    parser.add_argument("--games-per-iter", type=int,   default=GAMES_PER_ITER,  dest="games_per_iter")
    parser.add_argument("--batch-size",     type=int,   default=BATCH_SIZE,      dest="batch_size")
    parser.add_argument("--buffer",         type=int,   default=BUFFER_SIZE)
    parser.add_argument("--lr",             type=float, default=LEARNING_RATE)
    parser.add_argument("--save-every",     type=int,   default=SAVE_EVERY,      dest="save_every")
    parser.add_argument("--train-steps",    type=int,   default=TRAIN_STEPS,     dest="train_steps")
    parser.add_argument("--workers",        type=int,   default=NUM_WORKERS)
    parser.add_argument("--fresh",          action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    multiprocessing.freeze_support()   # needed for Windows frozen executables
    main()
