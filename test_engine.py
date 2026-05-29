"""
Smoke tests for the Scrabble engine (no ML).

Run:  python test_engine.py
"""

from board import Board, PREMIUM
from scoring import score_move
from bag import Bag
from gaddag import GADDAG
from move_generator import generate_moves

# ── Tiny word list for testing ──────────────────────────────────────────
WORDS = [
    "CAR", "CARE", "CARD", "SCAR", "CARS",
    "ARC",  "ARCS",
    "AT", "CAT", "CATS", "SAT", "TAR", "TARS",
    "HI", "HIM", "HIS",
    "IT", "SIT", "SITS",
    "OR", "FOR", "FORE",
    "SCORE", "SCARES", "SCORED",
]


def make_gaddag() -> GADDAG:
    g = GADDAG()
    g.build_from_wordlist(WORDS)
    return g


def test_gaddag_contains() -> None:
    g = make_gaddag()
    assert g.contains("CARE"), "CARE should be in GADDAG"
    assert g.contains("CAR"),  "CAR should be in GADDAG"
    assert not g.contains("DOG"), "DOG should NOT be in GADDAG"
    assert not g.contains("CA"),  "CA should NOT be in GADDAG"
    print("PASS  test_gaddag_contains")


def test_premium_symmetry() -> None:
    # TW at (0,0) should mirror to (0,14), (14,0), (14,14)
    for r, c in [(0, 0), (0, 14), (14, 0), (14, 14)]:
        assert PREMIUM.get((r, c)) == "TW", f"Expected TW at ({r},{c})"
    # Centre (7,7) = DW (star)
    assert PREMIUM.get((7, 7)) == "DW"
    print("PASS  test_premium_symmetry")


def test_bag_draw() -> None:
    bag = Bag(seed=42)
    assert len(bag) == 100
    tiles = bag.draw(7)
    assert len(tiles) == 7
    assert len(bag) == 93
    print("PASS  test_bag_draw")


def test_score_simple() -> None:
    """Place 'CAR' horizontally at row 7, col 7 (centre DW square)."""
    board = Board()
    placements = [
        (7, 7, "C", False),
        (7, 8, "A", False),
        (7, 9, "R", False),
    ]
    for r, c, letter, blank in placements:
        board.place_tile(r, c, letter, blank)

    pts = score_move(board, placements, horizontal=True)
    # C=3, A=1, R=1 = 5; DW at (7,7) → 10
    assert pts == 10, f"Expected 10, got {pts}"
    print("PASS  test_score_simple")


def test_move_generation_first_move() -> None:
    """On an empty board with rack [C,A,R,S,E,T,I], we should find ≥1 move."""
    g = make_gaddag()
    board = Board()
    rack = list("CARSET I".replace(" ", ""))  # C A R S E T I
    rack = ["C", "A", "R", "S", "E", "T", "I"]

    moves = generate_moves(board, g, rack)
    # Score them
    for m in moves:
        for r, c, letter, is_blank in m.placements:
            board.place_tile(r, c, letter, is_blank)
        from scoring import score_move as _sm
        m.score = _sm(board, m.placements, m.is_horizontal)
        for r, c, _, _ in m.placements:
            board.remove_tile(r, c)

    print(f"  Found {len(moves)} legal moves on first turn")
    if moves:
        best = max(moves, key=lambda m: m.score)
        print(f"  Best: {best}")
    assert len(moves) >= 1, "Should find at least one move"
    print("PASS  test_move_generation_first_move")


if __name__ == "__main__":
    test_gaddag_contains()
    test_premium_symmetry()
    test_bag_draw()
    test_score_simple()
    test_move_generation_first_move()
    print("\nAll tests passed.")
