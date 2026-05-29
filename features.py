"""
Feature extraction — converts game state into tensors for the neural network.

We have three separate input streams that the model combines:

  1. Board tensor  [27 x 15 x 15]
       One binary channel per letter (A-Z + blank).
       Channel k = 1 wherever letter k appears on the board.
       This is passed through a CNN to capture spatial patterns.

  2. Rack vector   [27]
       Normalised count of each letter in the current rack.
       (count / RACK_SIZE, so values are in [0, 1])

  3. Move features [N_MOVE_FEATURES]
       Hand-crafted scalars that describe a specific candidate move.
       Chosen so the model can learn strategic concepts like:
         - Is this move a bingo?
         - How good are the tiles I'm keeping?
         - Am I opening a Triple Word score for my opponent?

Why three streams?
  The board is spatial data — a CNN is the right tool.
  The rack and move scalars are tabular — an MLP handles those.
  Combining them in a "late fusion" head lets each stream specialise.

Leave value
  After placing tiles, the letters left in your rack are called the "leave".
  A balanced leave (mix of vowels/consonants, keeping S and blanks) is
  worth more than the raw score suggests.  We use a precomputed table of
  single-tile leave values; the leave score is the sum over remaining tiles.

  This is one of the most important features in real Scrabble AI.
"""

from __future__ import annotations
import torch
import numpy as np
from board import Board, TILE_SCORES
from move_generator import Move

# ── Constants ────────────────────────────────────────────────────────────────

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ?"   # 27 symbols (? = blank)
LETTER_INDEX = {ch: i for i, ch in enumerate(LETTERS)}

BOARD_CHANNELS = 27      # one per letter + blank
BOARD_SIZE     = 15
RACK_SIZE      = 7

# ── Leave values ─────────────────────────────────────────────────────────────
# Estimated strategic value of keeping each tile in your rack (beyond face value).
# Derived from empirical Scrabble analysis (simplified version of Quackle's values).
# Positive = good to keep, negative = bad to keep.
LEAVE_VALUES: dict[str, float] = {
    "A":  0.6, "B": -1.5, "C": -1.2, "D": -0.8, "E":  1.2,
    "F": -1.8, "G": -1.5, "H": -0.5, "I":  0.5, "J": -3.0,
    "K": -0.8, "L":  0.5, "M": -0.8, "N":  0.5, "O":  0.2,
    "P": -1.5, "Q": -5.0, "R":  0.8, "S":  3.5, "T":  0.5,
    "U": -0.5, "V": -2.5, "W": -1.5, "X": -0.5, "Y": -0.5,
    "Z": -1.5, "?":  8.0,  # blank is extremely valuable
}

# Penalty for duplicate tiles in the leave (hurts flexibility)
DUPLICATE_PENALTY = -1.0

# Number of scalar move features
N_MOVE_FEATURES = 14


# ── Board tensor ─────────────────────────────────────────────────────────────

def board_to_tensor(board: Board) -> torch.Tensor:
    """
    Returns a float32 tensor of shape [27, 15, 15].

    Channel i is 1.0 at every square occupied by letter LETTERS[i], 0 elsewhere.
    Blank tiles contribute to channel 26 ("?" channel), not their display letter.
    """
    t = torch.zeros(BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            sq = board.get(r, c)
            if sq.tile is not None:
                if sq.is_blank:
                    t[26, r, c] = 1.0        # blank channel
                else:
                    idx = LETTER_INDEX.get(sq.tile, -1)
                    if idx >= 0:
                        t[idx, r, c] = 1.0
    return t


# ── Rack vector ───────────────────────────────────────────────────────────────

def rack_to_tensor(rack: list[str]) -> torch.Tensor:
    """
    Returns a float32 tensor of shape [27].
    Each element = count of that letter / RACK_SIZE  (normalised to [0,1]).
    """
    t = torch.zeros(BOARD_CHANNELS, dtype=torch.float32)
    for tile in rack:
        idx = LETTER_INDEX.get(tile.upper(), -1)
        if idx >= 0:
            t[idx] += 1.0
    return t / RACK_SIZE


# ── Leave quality ─────────────────────────────────────────────────────────────

def leave_score(leave: list[str]) -> float:
    """
    Estimate the strategic value of the tiles remaining in the rack after a move.
    Higher = better rack for future turns.
    """
    score = sum(LEAVE_VALUES.get(t.upper(), 0.0) for t in leave)
    # Penalise duplicate letters (hurts bingo chances)
    counts: dict[str, int] = {}
    for t in leave:
        counts[t] = counts.get(t, 0) + 1
    for cnt in counts.values():
        if cnt > 1:
            score += DUPLICATE_PENALTY * (cnt - 1)
    return score


# ── Move feature vector ───────────────────────────────────────────────────────

def move_features(
    move: Move,
    rack: list[str],
    bag_size: int,
    board: Board,
) -> torch.Tensor:
    """
    Returns a float32 tensor of shape [N_MOVE_FEATURES].

    Features (all normalised to roughly [-1, 1] or [0, 1]):

    0   raw score / 100
    1   number of tiles placed / 7
    2   is bingo (1 if all 7 tiles used, else 0)
    3   leave score / 10
    4   combined value (score + leave) / 100
    5   anchor column / 14           (left=0, right=1)
    6   anchor row / 14              (top=0, bottom=1)
    7   distance from centre / 10   (centre=0, corner=~10)
    8   word length / 15
    9   1 if horizontal, 0 if vertical
    10  number of premium squares used by NEW tiles / 7
    11  does the move open a TW square for the opponent? (0/1)
    12  bag size / 100
    13  score of highest-value tile played / 10
    """
    n_placed = len(move.placements)
    is_bingo = float(n_placed == 7)

    # Compute leave (tiles in rack that weren't played)
    played_tiles: list[str] = []
    for _, _, letter, is_blank in move.placements:
        played_tiles.append("?" if is_blank else letter)
    remaining = list(rack)
    for t in played_tiles:
        if t in remaining:
            remaining.remove(t)
    lv = leave_score(remaining)

    raw = move.score
    combined = raw + lv

    # Position features
    r0, c0 = move.row, move.col
    centre = 7.0
    dist_centre = ((r0 - centre) ** 2 + (c0 - centre) ** 2) ** 0.5

    # Premium squares hit by new tiles
    from board import PREMIUM
    premiums_hit = sum(
        1 for r, c, _, _ in move.placements
        if PREMIUM.get((r, c)) is not None
    )

    # Does this move leave a TW square adjacent and exposed for the opponent?
    tw_exposed = _opens_tw(board, move)

    # Highest scoring tile played
    max_tile_score = max(
        (TILE_SCORES.get(l, 0) for _, _, l, b in move.placements if not b),
        default=0,
    )

    feats = [
        raw / 100.0,
        n_placed / 7.0,
        is_bingo,
        lv / 10.0,
        combined / 100.0,
        c0 / 14.0,
        r0 / 14.0,
        dist_centre / 10.0,
        len(move.word) / 15.0,
        float(move.is_horizontal),
        premiums_hit / 7.0,
        float(tw_exposed),
        bag_size / 100.0,
        max_tile_score / 10.0,
    ]
    assert len(feats) == N_MOVE_FEATURES
    return torch.tensor(feats, dtype=torch.float32)


def _opens_tw(board: Board, move: Move) -> bool:
    """
    Returns True if any newly placed tile is directly adjacent (horizontally or
    vertically) to an unoccupied Triple Word square — i.e. we're leaving a TW
    open for the opponent to exploit.
    """
    from board import PREMIUM
    for r, c, _, _ in move.placements:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (0 <= nr < 15 and 0 <= nc < 15
                    and PREMIUM.get((nr, nc)) == "TW"
                    and board.get(nr, nc).tile is None):
                return True
    return False
