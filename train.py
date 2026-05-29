"""
Training loop for ScrabbleNet.

Overview
────────
We use a replay buffer — a fixed-size pool of ExperienceTuples.  Each
training iteration:

  1. Play N self-play games to generate new experience.
  2. Add experience to the replay buffer (oldest entries dropped when full).
  3. Sample a random minibatch from the buffer.
  4. Compute MSE loss between model prediction and discounted target.
  5. Backprop + Adam optimiser step.
  6. Decay epsilon (less exploration over time).
  7. Save a checkpoint every K iterations.

Replay buffer
─────────────
Why not just train on the most recent games?
  Recent games come from a correlated sequence of states.  Training on
  correlated data causes the network to forget earlier patterns (catastrophic
  forgetting).  Sampling randomly from a large buffer breaks the correlation,
  which stabilises training — this was a key insight in DeepMind's DQN (2015).

Loss function: MSE
──────────────────
Our target is a continuous scalar (score margin), so Mean Squared Error is
the natural choice.  If we were classifying moves as "good/bad" we'd use
cross-entropy, but regression with MSE fits what we're doing.

Adam optimiser
──────────────
Stochastic Gradient Descent (SGD) + momentum + per-parameter learning rates.
Adam almost always converges faster than plain SGD on problems like this.
lr=1e-3 is a standard starting point; we add weight_decay (L2 regularisation)
to prevent overfitting when experience is limited.

Run:  python train.py
      python train.py --iterations 200 --games-per-iter 20 --batch-size 256
"""

import argparse
import collections
import random
import time
import pathlib

import torch
import torch.nn as nn
import torch.optim as optim

from gaddag import GADDAG
from model import ScrabbleNet
from selfplay import play_game, ExperienceTuple

# ── Defaults ─────────────────────────────────────────────────────────────────

DICT_PATH     = "TWL06.txt"
CACHE_PATH    = "gaddag.pkl"
CHECKPOINT_DIR = pathlib.Path("checkpoints")

BUFFER_SIZE     = 50_000   # max experience tuples in replay buffer
BATCH_SIZE      = 256
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-4
GAMES_PER_ITER  = 10       # self-play games generated per training iteration
TRAIN_STEPS     = 4        # gradient steps per iteration
ITERATIONS      = 100
EPSILON_START   = 1.0
EPSILON_END     = 0.05
SAVE_EVERY      = 10       # save checkpoint every N iterations


def build_batch(
    samples: list[ExperienceTuple],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack a list of ExperienceTuples into GPU-ready batched tensors."""
    boards  = torch.stack([s.board_t for s in samples]).to(device)
    racks   = torch.stack([s.rack_t  for s in samples]).to(device)
    moves   = torch.stack([s.move_f  for s in samples]).to(device)
    targets = torch.tensor([s.target for s in samples],
                           dtype=torch.float32, device=device).unsqueeze(1)
    return boards, racks, moves, targets


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load dictionary ───────────────────────────────────────────────
    gaddag = GADDAG.load_or_build(args.dict, args.cache)
    print(f"Dictionary loaded.  Testing: QUETZAL={gaddag.contains('QUETZAL')}")

    # ── Model + optimiser ─────────────────────────────────────────────
    model = ScrabbleNet().to(device)
    print(f"ScrabbleNet parameters: {model.parameter_count():,}")

    # Load checkpoint if resuming
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

    # ── Replay buffer ─────────────────────────────────────────────────
    buffer: collections.deque[ExperienceTuple] = collections.deque(maxlen=args.buffer)

    # ── Epsilon schedule ──────────────────────────────────────────────
    # Linear decay from EPSILON_START to EPSILON_END over all iterations
    def epsilon_at(iteration: int) -> float:
        frac = min(iteration / max(args.iterations - 1, 1), 1.0)
        return EPSILON_START + frac * (EPSILON_END - EPSILON_START)

    # ── Training loop ─────────────────────────────────────────────────
    loss_history: list[float] = []

    for it in range(start_iter, start_iter + args.iterations):
        eps = epsilon_at(it - start_iter)
        t0  = time.time()

        # 1. Generate self-play experience
        new_exp = 0
        for _ in range(args.games_per_iter):
            tuples = play_game(gaddag, model=model, epsilon=eps)
            buffer.extend(tuples)
            new_exp += len(tuples)

        # 2. Train (only once we have enough data for a batch)
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
                # Gradient clipping — prevents exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                iter_losses.append(loss.item())

        avg_loss = sum(iter_losses) / len(iter_losses) if iter_losses else float("nan")
        loss_history.append(avg_loss)
        elapsed = time.time() - t0

        print(
            f"[iter {it+1:04d}]  eps={eps:.3f}  "
            f"buf={len(buffer):>6,}  +exp={new_exp:>4}  "
            f"loss={avg_loss:.4f}  ({elapsed:.1f}s)"
        )

        # 3. Save checkpoint
        if (it + 1) % args.save_every == 0:
            ckpt_path = CHECKPOINT_DIR / f"ckpt_{it+1:04d}.pt"
            torch.save({"model": model.state_dict(), "iteration": it + 1}, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    print("\nTraining complete.")

    # ── Save final model ──────────────────────────────────────────────
    final_path = CHECKPOINT_DIR / "model_final.pt"
    torch.save({"model": model.state_dict(), "iteration": start_iter + args.iterations}, final_path)
    print(f"Final model saved to {final_path}")

    # ── Print simple loss summary ─────────────────────────────────────
    valid = [l for l in loss_history if l == l]  # filter NaN
    if valid:
        n = len(valid)
        first_q = valid[:n // 4]
        last_q  = valid[-(n // 4) or 1:]
        print(f"\nLoss summary:")
        print(f"  First quarter avg : {sum(first_q)/len(first_q):.4f}")
        print(f"  Last quarter avg  : {sum(last_q)/len(last_q):.4f}")


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
    parser.add_argument("--fresh",          action="store_true",
                        help="Ignore existing checkpoints and start fresh.")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
