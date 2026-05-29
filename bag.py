"""Tile bag — draw and return tiles."""

import random
from board import TILE_DISTRIBUTION


class Bag:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._tiles: list[str] = []
        for letter, count in TILE_DISTRIBUTION.items():
            self._tiles.extend([letter] * count)
        self._rng.shuffle(self._tiles)

    def __len__(self) -> int:
        return len(self._tiles)

    def draw(self, n: int = 1) -> list[str]:
        """Draw up to n tiles (fewer if bag runs low)."""
        n = min(n, len(self._tiles))
        drawn = self._tiles[-n:]
        self._tiles = self._tiles[:-n]
        return drawn

    def return_tiles(self, tiles: list[str]) -> None:
        """Put tiles back and reshuffle (used for exchanges)."""
        self._tiles.extend(tiles)
        self._rng.shuffle(self._tiles)

    def is_empty(self) -> bool:
        return len(self._tiles) == 0
