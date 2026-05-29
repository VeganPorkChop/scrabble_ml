"""
Play a game against the Scrabble AI.

Usage
-----
    python play.py                          # greedy AI (no model needed)
    python play.py --model checkpoints/model_final.pt
    python play.py --model checkpoints/model_final.pt --ai-first

Commands (your turn)
--------------------
    WORD ROW COL H      place word horizontally  (e.g.  CARE 7 7 H)
    WORD ROW COL V      place word vertically     (e.g.  CARE 7 7 V)
    PASS                skip your turn
    EXCHANGE ABC        swap tiles A, B, C back into the bag
    QUIT                resign

Row and column are 0-indexed (0–14).  Row 7, col 7 is the centre star.

The AI can play in two modes:
  - Greedy (no model): always picks the highest raw-score move.
  - Trained model: uses ScrabbleNet to evaluate moves (better strategy).
"""

import argparse
import pathlib
import sys
import torch

from gaddag import GADDAG
from board import Board
from bag import Bag
from move_generator import generate_moves, Move
from scoring import score_move as _score_move
from features import board_to_tensor, rack_to_tensor, move_features

RACK_SIZE = 7
DICT_PATH = "TWL06.txt"
CACHE_PATH = "gaddag.pkl"


# ── Board display ─────────────────────────────────────────────────────────────

def display(board: Board, human_rack: list[str], ai_rack_size: int,
            human_score: int, ai_score: int, bag_size: int) -> None:
    print("\n" + board.display())
    print(f"\n  You : {human_score} pts  |  rack : {' '.join(sorted(human_rack))}")
    print(f"  AI  : {ai_score} pts  |  rack : {ai_rack_size} tiles")
    print(f"  Bag : {bag_size} tiles remaining\n")


# ── AI move selection ─────────────────────────────────────────────────────────

def ai_choose(board: Board, rack: list[str], bag: Bag,
              gaddag: GADDAG, model) -> Move | None:
    """Pick the best move for the AI. Returns None if no legal moves."""
    moves = generate_moves(board, gaddag, rack)
    if not moves:
        return None

    # Score all moves
    for m in moves:
        for r, c, l, b in m.placements:
            board.place_tile(r, c, l, b)
        m.score = _score_move(board, m.placements, m.is_horizontal)
        for r, c, _, _ in m.placements:
            board.remove_tile(r, c)

    if model is None:
        # Greedy: best raw score
        return max(moves, key=lambda m: m.score)

    # Neural network: batch-evaluate all candidates
    model.eval()
    board_t = board_to_tensor(board)
    rack_t  = rack_to_tensor(rack)
    with torch.no_grad():
        move_feats = torch.stack([move_features(m, rack, len(bag), board) for m in moves])
        n = len(moves)
        boards_b = board_t.unsqueeze(0).expand(n, -1, -1, -1)
        racks_b  = rack_t.unsqueeze(0).expand(n, -1)
        vals = model(boards_b, racks_b, move_feats).squeeze(1)
    return moves[vals.argmax().item()]


# ── Human move parsing ────────────────────────────────────────────────────────

def parse_move(cmd: str, board: Board, rack: list[str],
               gaddag: GADDAG) -> Move | str:
    """
    Parse a move string like 'CARE 7 7 H'.
    Returns a Move on success, or an error string on failure.
    """
    parts = cmd.strip().upper().split()
    if len(parts) != 4:
        return "Format: WORD ROW COL H/V  (e.g.  CARE 7 7 H)"

    word, row_s, col_s, direction = parts
    try:
        row, col = int(row_s), int(col_s)
    except ValueError:
        return "ROW and COL must be integers 0-14."

    if not (0 <= row <= 14 and 0 <= col <= 14):
        return "ROW and COL must be between 0 and 14."

    if direction not in ("H", "V"):
        return "Direction must be H (horizontal) or V (vertical)."

    horizontal = direction == "H"
    dr, dc = (0, 1) if horizontal else (1, 0)

    # Build placements — match word letters to board / rack
    rack_copy = list(rack)
    placements: list[tuple[int, int, str, bool]] = []

    for i, letter in enumerate(word):
        r, c = row + i * dr, col + i * dc
        if not (0 <= r <= 14 and 0 <= c <= 14):
            return f"Word goes off the board at square ({r},{c})."
        sq = board.get(r, c)
        if sq.tile is not None:
            if sq.tile != letter:
                return (f"Square ({r},{c}) has '{sq.tile}' on it, "
                        f"but word needs '{letter}' there.")
            # Use existing tile — no rack tile consumed
        else:
            # Need a tile from the rack
            if letter in rack_copy:
                rack_copy.remove(letter)
                placements.append((r, c, letter, False))
            elif "?" in rack_copy:
                rack_copy.remove("?")
                placements.append((r, c, letter, True))   # blank used as letter
            else:
                return f"You don't have a '{letter}' or blank in your rack."

    if not placements:
        return "No new tiles placed — the word is already fully on the board."

    # Validate all formed words via the GADDAG
    # Temporarily place tiles to check
    for r, c, letter, is_blank in placements:
        board.place_tile(r, c, letter, is_blank)

    # Collect the main word and any cross-words
    from scoring import _collect_word
    words_to_check: list[str] = []
    cells = _collect_word(board, row, col, horizontal)
    words_to_check.append("".join(board.get(r, c).tile for r, c in cells))  # type: ignore

    for r, c, _, _ in placements:
        cross = _collect_word(board, r, c, not horizontal)
        if len(cross) > 1:
            words_to_check.append("".join(board.get(rr, cc).tile for rr, cc in cross))  # type: ignore

    # Undo temporary placement
    for r, c, _, _ in placements:
        board.remove_tile(r, c)

    invalid = [w for w in words_to_check if not gaddag.contains(w)]
    if invalid:
        return f"Invalid word(s): {', '.join(invalid)}"

    # Board connectivity check (must touch existing tile unless first move)
    if board.is_empty():
        centre = (7, 7)
        if not any(r == centre[0] and c == centre[1] for r, c, _, _ in placements):
            return "First move must cover the centre square (7,7)."
    else:
        touches = any(board.has_adjacent_tile(r, c) or board.get(r, c).tile is not None
                      for r, c, _, _ in placements)
        if not touches:
            return "Word must connect to tiles already on the board."

    pts = 0
    for r, c, letter, is_blank in placements:
        board.place_tile(r, c, letter, is_blank)
    pts = _score_move(board, placements, horizontal)
    for r, c, _, _ in placements:
        board.remove_tile(r, c)

    return Move(
        word="".join(board.get(row + i*dr, col + i*dc).tile or word[i]
                     for i in range(len(word))),
        placements=placements,
        row=row, col=col,
        is_horizontal=horizontal,
        score=pts,
    )


# ── Main game loop ────────────────────────────────────────────────────────────

def play(args: argparse.Namespace) -> None:
    print("Loading dictionary...")
    gaddag = GADDAG.load_or_build(DICT_PATH, CACHE_PATH)

    model = None
    if args.model:
        model_path = pathlib.Path(args.model)
        if not model_path.exists():
            print(f"Model file not found: {model_path}")
            print("Falling back to greedy AI.")
        else:
            from model import ScrabbleNet
            net = ScrabbleNet()
            ckpt = torch.load(model_path, map_location="cpu")
            net.load_state_dict(ckpt["model"])
            net.eval()
            model = net
            it = ckpt.get("iteration", "?")
            print(f"Loaded model from {model_path}  (iteration {it})")

    board = Board()
    bag   = Bag()

    if args.ai_first:
        ai_rack, human_rack = bag.draw(RACK_SIZE), bag.draw(RACK_SIZE)
    else:
        human_rack, ai_rack = bag.draw(RACK_SIZE), bag.draw(RACK_SIZE)

    human_score = 0
    ai_score    = 0
    human_passes = 0
    ai_passes    = 0
    human_turn = not args.ai_first

    print("\n=== SCRABBLE ===")
    print("Type HELP for commands.\n")

    while True:
        display(board, human_rack, len(ai_rack), human_score, ai_score, len(bag))

        # ── End-game check ─────────────────────────────────────────────
        if human_passes >= 3 and ai_passes >= 3:
            print("Six consecutive passes — game over.")
            break
        if not human_rack and bag.is_empty():
            # Human went out
            penalty = sum(__import__("board").TILE_SCORES.get(t, 0)
                          for t in ai_rack if t != "?")
            ai_score    -= penalty
            human_score += penalty
            print("You've used all your tiles! Game over.")
            break
        if not ai_rack and bag.is_empty():
            penalty = sum(__import__("board").TILE_SCORES.get(t, 0)
                          for t in human_rack if t != "?")
            human_score -= penalty
            ai_score    += penalty
            print("AI used all its tiles! Game over.")
            break

        # ── Human turn ─────────────────────────────────────────────────
        if human_turn:
            while True:
                try:
                    cmd = input("Your move: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nQuitting.")
                    sys.exit(0)

                upper = cmd.upper()

                if upper == "QUIT":
                    print("You resigned.")
                    sys.exit(0)

                if upper == "PASS":
                    human_passes += 1
                    print("You passed.")
                    break

                if upper == "HELP":
                    print("  WORD ROW COL H/V   — place a word  (e.g. CARE 7 7 H)")
                    print("  PASS               — skip turn")
                    print("  EXCHANGE XYZ       — swap tiles X, Y, Z")
                    print("  QUIT               — resign")
                    continue

                if upper.startswith("EXCHANGE "):
                    tiles = list(upper[9:].replace(" ", ""))
                    if len(bag) < RACK_SIZE:
                        print("Can't exchange — fewer than 7 tiles in bag.")
                        continue
                    missing = []
                    rack_copy = list(human_rack)
                    for t in tiles:
                        if t in rack_copy:
                            rack_copy.remove(t)
                        else:
                            missing.append(t)
                    if missing:
                        print(f"You don't have: {' '.join(missing)}")
                        continue
                    for t in tiles:
                        human_rack.remove(t)
                    drawn = bag.draw(len(tiles))
                    bag.return_tiles(tiles)
                    human_rack.extend(drawn)
                    human_passes += 1
                    print(f"Exchanged {len(tiles)} tile(s).  New rack: {' '.join(sorted(human_rack))}")
                    break

                result = parse_move(cmd, board, human_rack, gaddag)
                if isinstance(result, str):
                    print(f"  Error: {result}")
                    continue

                # Apply move
                for r, c, letter, is_blank in result.placements:
                    board.place_tile(r, c, letter, is_blank)
                human_score += result.score
                for _, _, letter, is_blank in result.placements:
                    human_rack.remove("?" if is_blank else letter)
                human_rack.extend(bag.draw(RACK_SIZE - len(human_rack)))
                human_passes = 0
                print(f"  You played {result.word!r} for {result.score} pts.")
                break

        # ── AI turn ────────────────────────────────────────────────────
        else:
            print("AI is thinking...")
            chosen = ai_choose(board, ai_rack, bag, gaddag, model)
            if chosen is None:
                ai_passes += 1
                print("AI passed.")
            else:
                for r, c, letter, is_blank in chosen.placements:
                    board.place_tile(r, c, letter, is_blank)
                ai_score += chosen.score
                for _, _, letter, is_blank in chosen.placements:
                    ai_rack.remove("?" if is_blank else letter)
                ai_rack.extend(bag.draw(RACK_SIZE - len(ai_rack)))
                ai_passes = 0
                direction = "H" if chosen.is_horizontal else "V"
                print(f"  AI played {chosen.word!r} at ({chosen.row},{chosen.col}) {direction} "
                      f"for {chosen.score} pts.")

        human_turn = not human_turn

    # ── Final scores ───────────────────────────────────────────────────
    print(f"\n=== FINAL SCORES ===")
    print(f"  You : {human_score}")
    print(f"  AI  : {ai_score}")
    if human_score > ai_score:
        print("  You win!")
    elif ai_score > human_score:
        print("  AI wins.")
    else:
        print("  It's a tie!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Scrabble against the AI.")
    parser.add_argument("--model",    default=None,
                        help="Path to a trained model checkpoint (.pt file).")
    parser.add_argument("--ai-first", action="store_true", dest="ai_first",
                        help="Let the AI go first.")
    args = parser.parse_args()
    play(args)


if __name__ == "__main__":
    main()
