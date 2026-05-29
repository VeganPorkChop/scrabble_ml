"""
Move generator — finds ALL legal moves for a given board + rack.

Algorithm
---------
This implements the Appel & Jacobson (1988) algorithm, adapted to use a GADDAG
instead of their original DAWG.  The key ideas:

1. **Anchors**: Any empty square that touches an existing tile is an "anchor".
   On an empty board the centre (7,7) is the only anchor.

2. **Cross-checks**: For each anchor, we precompute which letters are allowed
   there by the *perpendicular* words they would form.  A letter is legal only
   if placing it there (combined with existing tiles above/below or left/right)
   forms a valid word in every perpendicular direction.

3. **GADDAG traversal**: Starting from each anchor we walk the GADDAG, filling
   squares to the *left* of the anchor (using the reversed-prefix part of the
   GADDAG path) then crossing the separator and filling squares to the *right*.

The result is a list of Move objects, each fully describing the placement.

Key types
---------
Rack       : dict[str, int]  — letter → count ("?" = blank)
Move       : dataclass with placements, word, score, is_horizontal
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from board import Board
    from gaddag import GADDAG

from gaddag import SEP


@dataclass
class Move:
    """A complete legal move."""
    word: str                               # full word as placed
    placements: list[tuple[int, int, str, bool]]  # (row, col, letter, is_blank)
    row: int                                # starting row
    col: int                                # starting col
    is_horizontal: bool
    score: int = 0                          # filled in by score_move()

    def __repr__(self) -> str:
        direction = "H" if self.is_horizontal else "V"
        return f"Move({self.word} @ ({self.row},{self.col}) {direction} score={self.score})"


# ──────────────────────────────────────────────────────────────────────
# Cross-check computation
# ──────────────────────────────────────────────────────────────────────

def _compute_cross_checks(
    board: "Board",
    gaddag: "GADDAG",
    horizontal: bool,
) -> dict[tuple[int, int], set[str]]:
    """
    For every empty square, compute the set of letters that may legally be
    placed there given the tiles already on the board in the perpendicular
    direction.

    If there are no perpendicular neighbours the square is unconstrained:
    we return the full alphabet.

    horizontal=True  → main word goes left-right, cross-words go up-down
    horizontal=False → main word goes top-bottom, cross-words go left-right
    """
    SIZE = board.SIZE
    ALL_LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    # perpendicular direction
    dr, dc = (1, 0) if horizontal else (0, 1)

    checks: dict[tuple[int, int], set[str]] = {}

    for row in range(SIZE):
        for col in range(SIZE):
            if board.get(row, col).tile is not None:
                continue  # occupied

            # Collect prefix (tiles above/left of this square in perp direction)
            prefix: list[str] = []
            r, c = row - dr, col - dc
            while 0 <= r < SIZE and 0 <= c < SIZE and board.get(r, c).tile:
                prefix.append(board.get(r, c).tile)  # type: ignore[arg-type]
                r -= dr
                c -= dc
            prefix.reverse()

            # Collect suffix (tiles below/right)
            suffix: list[str] = []
            r, c = row + dr, col + dc
            while 0 <= r < SIZE and 0 <= c < SIZE and board.get(r, c).tile:
                suffix.append(board.get(r, c).tile)  # type: ignore[arg-type]
                r += dr
                c += dc

            if not prefix and not suffix:
                checks[(row, col)] = ALL_LETTERS  # unconstrained
                continue

            # Try each letter: does prefix + letter + suffix form a valid word?
            legal: set[str] = set()
            for letter in ALL_LETTERS:
                candidate = "".join(prefix) + letter + "".join(suffix)
                if gaddag.contains(candidate):
                    legal.add(letter)
            checks[(row, col)] = legal

    return checks


# ──────────────────────────────────────────────────────────────────────
# Anchor squares
# ──────────────────────────────────────────────────────────────────────

def _get_anchors(board: "Board") -> list[tuple[int, int]]:
    """Return all empty squares adjacent to an occupied square (or the centre on an empty board)."""
    if board.is_empty():
        return [board.CENTER]
    anchors = []
    SIZE = board.SIZE
    for r in range(SIZE):
        for c in range(SIZE):
            if board.get(r, c).tile is None and board.has_adjacent_tile(r, c):
                anchors.append((r, c))
    return anchors


# ──────────────────────────────────────────────────────────────────────
# GADDAG traversal helpers
# ──────────────────────────────────────────────────────────────────────

def _rack_to_dict(rack: list[str]) -> dict[str, int]:
    d: dict[str, int] = {}
    for t in rack:
        d[t] = d.get(t, 0) + 1
    return d


def generate_moves(
    board: "Board",
    gaddag: "GADDAG",
    rack: list[str],
) -> list[Move]:
    """
    Generate all legal moves for the given rack on the current board.

    Returns a list of Move objects (score field is 0 — caller should score them).
    """
    moves: list[Move] = []

    for horizontal in (True, False):
        cross_checks = _compute_cross_checks(board, gaddag, horizontal)
        anchors = _get_anchors(board)
        rack_dict = _rack_to_dict(rack)

        for anchor_r, anchor_c in anchors:
            _gen_from_anchor(
                board, gaddag, rack_dict, cross_checks,
                anchor_r, anchor_c, horizontal, moves,
            )

    # Deduplicate (same word/position can be found via different anchor paths)
    seen: set[tuple] = set()
    unique: list[Move] = []
    for m in moves:
        key = (m.row, m.col, m.word, m.is_horizontal)
        if key not in seen:
            seen.add(key)
            unique.append(m)
    return unique


# ──────────────────────────────────────────────────────────────────────
# Core recursive traversal
# ──────────────────────────────────────────────────────────────────────

def _gen_from_anchor(
    board: "Board",
    gaddag: "GADDAG",
    rack: dict[str, int],
    cross_checks: dict[tuple[int, int], set[str]],
    anchor_r: int,
    anchor_c: int,
    horizontal: bool,
    results: list[Move],
) -> None:
    """
    Enumerate all words that can be placed with (anchor_r, anchor_c) as one
    of their squares.  Uses a depth-first traversal of the GADDAG.

    State threaded through the recursion:
      node       : current GADDAG node
      rack       : mutable dict, tiles remaining (modified + restored)
      left_part  : letters collected going *left* of the anchor
      placements : (row, col, letter, is_blank) for tiles we've placed
      cur_r/c    : current square being filled
      going_left : True while building the reversed prefix, False after SEP
    """

    dr = 0 if horizontal else 1
    dc = 1 if horizontal else 0

    def _try_letter(
        node,
        r: int,
        c: int,
        going_left: bool,
        left_letters: list[str],
        right_letters: list[str],
        placements: list[tuple[int, int, str, bool]],
    ) -> None:
        SIZE = board.SIZE
        sq = board.get(r, c)

        if sq.tile is not None:
            # Square already occupied — we must use that tile
            letter = sq.tile
            child = node.child(letter)
            if child is None:
                return
            if going_left:
                left_letters.append(letter)
                _continue(child, r, c, going_left, left_letters, right_letters, placements)
                left_letters.pop()
            else:
                right_letters.append(letter)
                _continue(child, r, c, going_left, left_letters, right_letters, placements)
                right_letters.pop()
            return

        # Empty square — try each tile from the rack
        letters_to_try: list[tuple[str, bool]] = []  # (board_letter, is_blank)
        if rack.get("?", 0) > 0:
            for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                letters_to_try.append((ch, True))
        for tile, cnt in rack.items():
            if tile != "?" and cnt > 0:
                letters_to_try.append((tile, False))

        for letter, is_blank in letters_to_try:
            # Cross-check: the letter must be legal for this square
            if letter not in cross_checks.get((r, c), set()):
                continue
            child = node.child(letter)
            if child is None:
                continue

            # Place the tile
            tile_key = "?" if is_blank else letter
            rack[tile_key] -= 1
            board.place_tile(r, c, letter, is_blank)
            placements.append((r, c, letter, is_blank))

            if going_left:
                left_letters.append(letter)
                _continue(child, r, c, going_left, left_letters, right_letters, placements)
                left_letters.pop()
            else:
                right_letters.append(letter)
                _continue(child, r, c, going_left, left_letters, right_letters, placements)
                right_letters.pop()

            # Undo
            board.remove_tile(r, c)
            placements.pop()
            rack[tile_key] += 1

    def _continue(
        node,
        r: int,
        c: int,
        going_left: bool,
        left_letters: list[str],
        right_letters: list[str],
        placements: list[tuple[int, int, str, bool]],
    ) -> None:
        SIZE = board.SIZE

        if going_left:
            # Can we cross the separator and start going right?
            sep_child = node.child(SEP)
            if sep_child is not None:
                # Try to record a word if the right side is already empty or
                # extends to an existing tile run
                _try_right_from_anchor(
                    sep_child, anchor_r, anchor_c,
                    left_letters, right_letters, placements,
                )

            # Continue leftward
            nr, nc = r - dr, c - dc
            if 0 <= nr < SIZE and 0 <= nc < SIZE:
                # Only extend left into unoccupied or occupied squares
                _try_letter(node, nr, nc, True, left_letters, right_letters, placements)

        else:
            # Going right — record word if this node is a word end and next sq is empty/OOB
            nr, nc = r + dr, c + dc
            out_of_bounds = not (0 <= nr < SIZE and 0 <= nc < SIZE)
            next_empty = out_of_bounds or board.get(nr, nc).tile is None

            if node.is_end and next_empty and placements:
                _record(left_letters, right_letters, placements)

            # Continue rightward
            if not out_of_bounds:
                _try_letter(node, nr, nc, False, left_letters, right_letters, placements)

    def _try_right_from_anchor(
        node,
        r: int,
        c: int,
        left_letters: list[str],
        right_letters: list[str],
        placements: list[tuple[int, int, str, bool]],
    ) -> None:
        """After crossing the SEP, fill squares to the right of the anchor."""
        SIZE = board.SIZE
        # Check if word can end right here (anchor was the last letter of the word)
        nr, nc = r + dr, c + dc
        out_of_bounds = not (0 <= nr < SIZE and 0 <= nc < SIZE)
        next_empty = out_of_bounds or board.get(nr, nc).tile is None

        if node.is_end and next_empty and placements:
            _record(left_letters, right_letters, placements)

        if not out_of_bounds:
            _try_letter(node, nr, nc, False, left_letters, right_letters, placements)

    def _record(
        left_letters: list[str],
        right_letters: list[str],
        placements: list[tuple[int, int, str, bool]],
    ) -> None:
        """A complete valid word has been found — record the move."""
        if not placements:
            return

        # Reconstruct the full word
        word = "".join(reversed(left_letters)) + "".join(right_letters)
        if len(word) < 2:
            return

        # The starting square is the leftmost/topmost tile placed or already on board
        all_rs = [r for r, c, _, _ in placements]
        all_cs = [c for r, c, _, _ in placements]
        start_r = min(all_rs) if horizontal else min(all_rs)
        start_c = min(all_cs) if horizontal else min(all_cs)

        # Walk from start to find the actual word start (include pre-existing tiles)
        r, c = start_r, start_c
        while 0 <= r - dr < board.SIZE and 0 <= c - dc < board.SIZE and board.get(r - dr, c - dc).tile:
            r -= dr
            c -= dc
        start_r, start_c = r, c

        results.append(Move(
            word=word,
            placements=list(placements),
            row=start_r,
            col=start_c,
            is_horizontal=horizontal,
            score=0,
        ))

    # ── Kick off from the anchor ──
    # The anchor is either an empty square (normal case) or we start from
    # a letter already on the board.  We try each GADDAG starting letter.
    _try_letter(gaddag.root, anchor_r, anchor_c, True, [], [], [])
