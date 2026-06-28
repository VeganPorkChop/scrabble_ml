"""
Self-play data generation.

Concept: Monte Carlo policy evaluation
───────────────────────────────────────
We can't hand-label "good" moves, so we generate our own training data by
having the engine play games against itself.

After each game we know who won and by how much.  We then label every
(board, rack, move) tuple from the winning player's turns with a POSITIVE
target, and every tuple from the losing player's turns with a NEGATIVE target.

More precisely, the target for a turn at time t is the discounted final margin:

    target = γ^(T-t) * final_margin

where γ ∈ (0,1] is a discount factor (we use 0.99).  This lets the network
learn "moves made later in the game mattered more to the outcome."

This is a simplified form of Monte Carlo Return estimation — a core idea in
reinforcement learning.

Move selection during self-play
────────────────────────────────
Early in training the network knows nothing, so we use an ε-greedy policy:
  - with probability ε: pick a RANDOM move (exploration)
  - with probability 1-ε: pick the move the network values highest (exploitation)

ε starts at 1.0 (pure random) and decays each game, so the network gradually
starts trusting its own judgement.

ExperienceTuple — what we store
────────────────────────────────
For each move played we store:
  board_t  : board tensor before the move       [27, 15, 15]
  rack_t   : rack tensor before the move        [27]
  move_f   : move feature vector                [N_MOVE_FEATURES]
  target   : discounted final margin (float)
"""

from __future__ import annotations
import random
import torch
from dataclasses import dataclass

from board import Board
from bag import Bag
from gaddag import GADDAG
from move_generator import Move, generate_moves
from scoring import score_move as _score_move
from features import board_to_tensor, rack_to_tensor, move_features

RACK_SIZE = 7
GAMMA = 0.99   # discount factor


@dataclass
class ExperienceTuple:
    board_t: torch.Tensor   # [27, 15, 15]
    rack_t:  torch.Tensor   # [27]
    move_f:  torch.Tensor   # [N_MOVE_FEATURES]
    target:  float          # discounted score margin


def _apply_move(board: Board, rack: list[str], bag: Bag, move: Move) -> int:
    """Place tiles, score, replenish rack. Returns points scored."""
    for r, c, letter, is_blank in move.placements:
        board.place_tile(r, c, letter, is_blank)
    pts = _score_move(board, move.placements, move.is_horizontal)
    for _, _, letter, is_blank in move.placements:
        tile = "?" if is_blank else letter
        rack.remove(tile)
    drawn = bag.draw(RACK_SIZE - len(rack))
    rack.extend(drawn)
    return pts


def play_game(
    gaddag: GADDAG,
    model=None,           # ScrabbleNet | None
    epsilon: float = 1.0, # exploration probability
    seed: int | None = None,
    verbose: bool = False,
) -> list[ExperienceTuple]:
    """
    Play one full self-play game.  Returns a list of ExperienceTuples.

    If model is None, moves are chosen greedily by raw score (good baseline).
    If model is provided, uses ε-greedy with the model's value estimate.
    """
    rng = random.Random(seed)
    board = Board()
    bag = Bag(seed=seed)

    racks = [bag.draw(RACK_SIZE), bag.draw(RACK_SIZE)]   # two players
    scores = [0, 0]
    passes = [0, 0]
    turn = 0

    # Per-turn records: (player_idx, board_t, rack_t, move_f)
    # We don't know the target until the game ends.
    records: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    max_turns = 200  # safety limit

    for _ in range(max_turns):
        player = turn % 2
        rack = racks[player]

        # Generate and score all legal moves
        moves = generate_moves(board, gaddag, rack)
        for m in moves:
            for r, c, l, b in m.placements:
                board.place_tile(r, c, l, b)
            m.score = _score_move(board, m.placements, m.is_horizontal)
            for r, c, _, _ in m.placements:
                board.remove_tile(r, c)

        if not moves:
            # No legal moves — pass
            passes[player] += 1
            if all(p >= 3 for p in passes):
                break   # six consecutive passes ends game
            turn += 1
            continue
        else:
            passes[player] = 0

        # Capture state BEFORE the move (needed for the experience tuple)
        board_t = board_to_tensor(board)
        rack_t  = rack_to_tensor(rack)

        # ── Move selection ────────────────────────────────────────
        if model is not None and rng.random() > epsilon:
            # Exploitation: score ALL candidates in ONE batched forward pass.
            # This is ~100x faster than calling score_move() in a loop because
            # PyTorch processes the whole batch on the GPU/CPU in parallel.
            model.eval()
            device = next(model.parameters()).device  # wherever the model lives (CPU or GPU)
            with torch.no_grad():
                move_feats = torch.stack([
                    move_features(m, rack, len(bag), board) for m in moves
                ]).to(device)                           # [N_moves, N_MOVE_FEATURES]
                n = len(moves)
                boards_b = board_t.unsqueeze(0).expand(n, -1, -1, -1).to(device)  # [N, 27, 15, 15]
                racks_b  = rack_t.unsqueeze(0).expand(n, -1).to(device)            # [N, 27]
                vals = model(boards_b, racks_b, move_feats).squeeze(1)             # [N]
            best_idx  = vals.argmax().item()
            chosen    = moves[best_idx]
        else:
            # Exploration: pick randomly (weighted by raw score so it plays
            # legal, sensible moves rather than single-letter duds)
            total = sum(max(m.score, 1) for m in moves)
            r_val = rng.uniform(0, total)
            cumulative = 0.0
            chosen = moves[-1]
            for m in moves:
                cumulative += max(m.score, 1)
                if cumulative >= r_val:
                    chosen = m
                    break

        # Record the experience
        mf = move_features(chosen, rack, len(bag), board)
        records.append((player, board_t, rack_t, mf))

        # Apply move
        pts = _apply_move(board, rack, bag, chosen)
        scores[player] += pts

        if verbose:
            print(f"P{player+1} plays {chosen.word!r} for {pts} pts (total {scores[player]})")

        # End condition: bag empty and a player empties rack
        if bag.is_empty() and not rack:
            # End-game tile adjustment
            for i, r in enumerate(racks):
                penalty = sum(
                    __import__("board").TILE_SCORES.get(t, 0) for t in r if t != "?"
                )
                scores[i] -= penalty
                scores[1 - i] += penalty
            break

        turn += 1

    # ── Assign discounted targets ─────────────────────────────────
    margin = scores[0] - scores[1]   # positive = player 0 won
    if verbose:
        print(f"Final scores: P1={scores[0]}  P2={scores[1]}  margin={margin:+d}")

    experiences: list[ExperienceTuple] = []
    T = len(records)
    for t_idx, (player, board_t, rack_t, mf) in enumerate(records):
        sign = 1.0 if player == 0 else -1.0
        discount = GAMMA ** (T - 1 - t_idx)
        target = sign * margin * discount / 100.0   # scale to ~[-1, 1]
        experiences.append(ExperienceTuple(board_t, rack_t, mf, target))

    return experiences
