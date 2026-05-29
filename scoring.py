"""
Move scoring.

A Move is a list of (row, col, letter, is_blank) placements.
score_move() handles:
  - letter premiums (DL, TL) on newly placed tiles
  - word premiums (DW, TW) accumulated across the word
  - cross-word scoring (all words formed, not just the main word)
  - 50-point bingo bonus for using all 7 tiles
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from board import Board

from board import TILE_SCORES, PREMIUM

# A single tile placement: (row, col, letter, is_blank)
Placement = tuple[int, int, str, bool]


def _score_word(
    board: "Board",
    cells: list[tuple[int, int]],      # ordered squares of the word
    new_squares: set[tuple[int, int]], # squares where tiles are being placed NOW
) -> int:
    """Score one word. Applies premiums only for newly placed tiles."""
    raw = 0
    word_mult = 1

    for r, c in cells:
        sq = board.get(r, c)
        letter = sq.tile
        assert letter is not None
        base = 0 if sq.is_blank else TILE_SCORES.get(letter, 0)

        if (r, c) in new_squares:
            prem = PREMIUM.get((r, c))
            if prem == "DL":
                base *= 2
            elif prem == "TL":
                base *= 3
            elif prem == "DW":
                word_mult *= 2
            elif prem == "TW":
                word_mult *= 3

        raw += base

    return raw * word_mult


def _collect_word(board: "Board", row: int, col: int, horizontal: bool) -> list[tuple[int, int]]:
    """Walk in both directions to collect all squares of the word containing (row,col)."""
    dr, dc = (0, 1) if horizontal else (1, 0)

    # Step backwards to start
    r, c = row, col
    while 0 <= r - dr < board.SIZE and 0 <= c - dc < board.SIZE and board.get(r - dr, c - dc).tile:
        r -= dr
        c -= dc

    cells = []
    while 0 <= r < board.SIZE and 0 <= c < board.SIZE and board.get(r, c).tile:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def score_move(
    board: "Board",
    placements: list[Placement],
    horizontal: bool,
) -> int:
    """
    Score a move after the tiles have already been placed on the board.

    Parameters
    ----------
    board       : Board with placements already applied.
    placements  : List of (row, col, letter, is_blank) for newly placed tiles.
    horizontal  : True if main word runs left→right, False if top→bottom.

    Returns
    -------
    Total score including all cross-words and bingo bonus.
    """
    new_squares = {(r, c) for r, c, _, _ in placements}
    total = 0
    scored_words: set[frozenset] = set()

    # Score the main word
    start_r, start_c = placements[0][0], placements[0][1]
    main_cells = _collect_word(board, start_r, start_c, horizontal)
    key = frozenset(main_cells)
    if len(main_cells) > 1 and key not in scored_words:
        total += _score_word(board, main_cells, new_squares)
        scored_words.add(key)

    # Score cross-words formed by each new tile
    for r, c, _, _ in placements:
        cross_cells = _collect_word(board, r, c, not horizontal)
        key = frozenset(cross_cells)
        if len(cross_cells) > 1 and key not in scored_words:
            total += _score_word(board, cross_cells, new_squares)
            scored_words.add(key)

    # 50-point bingo bonus for using all 7 rack tiles
    if len(placements) == 7:
        total += 50

    return total
