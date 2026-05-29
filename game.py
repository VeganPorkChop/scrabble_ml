"""
Game — ties together Board, Bag, GADDAG, move generation, and scoring.

A Game object manages:
  - Two players (human or AI)
  - Turn flow
  - Pass / exchange / resign actions
  - End-game tile penalty
"""

from __future__ import annotations
from dataclasses import dataclass, field

from board import Board
from bag import Bag
from gaddag import GADDAG
from move_generator import Move, generate_moves
from scoring import score_move

RACK_SIZE = 7


@dataclass
class Player:
    name: str
    rack: list[str] = field(default_factory=list)
    score: int = 0
    passes: int = 0


class Game:
    def __init__(self, gaddag: GADDAG, seed: int | None = None) -> None:
        self.board = Board()
        self.bag = Bag(seed=seed)
        self.gaddag = gaddag
        self.players: list[Player] = [Player("Player 1"), Player("Player 2")]
        self.turn = 0          # index into self.players
        self.moves_played = 0
        self.game_over = False

        # Deal initial racks
        for p in self.players:
            p.rack = self.bag.draw(RACK_SIZE)

    # ── Accessors ──────────────────────────────────────────────────────

    @property
    def current_player(self) -> Player:
        return self.players[self.turn]

    def legal_moves(self) -> list[Move]:
        """Return all legal moves for the current player, scored."""
        moves = generate_moves(self.board, self.gaddag, self.current_player.rack)
        for m in moves:
            # Temporarily apply placements to score
            for r, c, letter, is_blank in m.placements:
                self.board.place_tile(r, c, letter, is_blank)
            m.score = score_move(self.board, m.placements, m.is_horizontal)
            for r, c, _, _ in m.placements:
                self.board.remove_tile(r, c)
        moves.sort(key=lambda m: m.score, reverse=True)
        return moves

    # ── Actions ────────────────────────────────────────────────────────

    def play_move(self, move: Move) -> int:
        """Apply a move. Returns points scored."""
        player = self.current_player
        rack = player.rack

        # Apply placements
        for r, c, letter, is_blank in move.placements:
            self.board.place_tile(r, c, letter, is_blank)

        # Score (board already has tiles placed)
        pts = score_move(self.board, move.placements, move.is_horizontal)
        player.score += pts
        player.passes = 0

        # Remove played tiles from rack
        for _, _, letter, is_blank in move.placements:
            tile = "?" if is_blank else letter
            rack.remove(tile)

        # Refill rack
        drawn = self.bag.draw(RACK_SIZE - len(rack))
        rack.extend(drawn)

        self.moves_played += 1
        self._check_game_over()
        self._advance_turn()
        return pts

    def pass_turn(self) -> None:
        player = self.current_player
        player.passes += 1
        # Six consecutive passes (3 per player) ends the game
        if all(p.passes >= 3 for p in self.players):
            self._end_game()
        self._advance_turn()

    def exchange(self, tiles: list[str]) -> bool:
        """Exchange tiles (only legal when bag has ≥7 tiles)."""
        if len(self.bag) < RACK_SIZE:
            return False
        player = self.current_player
        for t in tiles:
            player.rack.remove(t)
        drawn = self.bag.draw(len(tiles))
        self.bag.return_tiles(tiles)
        player.rack.extend(drawn)
        player.passes += 1
        self._advance_turn()
        return True

    # ── Internal ───────────────────────────────────────────────────────

    def _advance_turn(self) -> None:
        self.turn = (self.turn + 1) % len(self.players)

    def _check_game_over(self) -> None:
        # Game ends when a player empties their rack and the bag is empty
        player = self.current_player
        if not player.rack and self.bag.is_empty():
            self._end_game(emptied_by=player)

    def _end_game(self, emptied_by: Player | None = None) -> None:
        self.game_over = True
        # Deduct unplayed tile values; if someone went out, they gain those points
        total_deducted = 0
        for p in self.players:
            if p is emptied_by:
                continue
            penalty = sum(
                (0 if t == "?" else __import__("board").TILE_SCORES.get(t, 0))
                for t in p.rack
            )
            p.score -= penalty
            total_deducted += penalty
        if emptied_by:
            emptied_by.score += total_deducted

    def status(self) -> str:
        lines = [self.board.display(), ""]
        for p in self.players:
            rack_str = " ".join(p.rack)
            lines.append(f"{p.name}: {p.score} pts | rack: [{rack_str}]")
        lines.append(f"Bag: {len(self.bag)} tiles remaining")
        if self.game_over:
            winner = max(self.players, key=lambda p: p.score)
            lines.append(f"\nGame over! Winner: {winner.name} ({winner.score} pts)")
        else:
            lines.append(f"\nCurrent turn: {self.current_player.name}")
        return "\n".join(lines)
