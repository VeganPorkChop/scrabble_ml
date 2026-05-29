"""
GADDAG — the data structure that makes Scrabble move generation fast.

Background
----------
A naive approach to finding valid Scrabble words tries every possible letter
combination, which is far too slow. The GADDAG (invented by Steven Gordon, 1994)
is a directed acyclic word graph that lets us traverse words in *both* directions
from any starting tile already on the board.

How it works
------------
For each word W = w1 w2 ... wn, we insert one path for every "hook" position k:

    w_k w_{k-1} ... w_1  <SEP>  w_{k+1} ... w_n

The separator character '<' marks the pivot — we traverse left-to-right up to
the hook, then the separator signals "now go right from the anchor".

So GADDAG = trie over all reversed-prefix + separator + suffix strings.

Example: "CARE"
  Inserted paths (using '<' as separator):
    C < A R E          (hook at position 0: nothing to left, go right)
    A C < R E          (hook at 1: left part reversed = "AC", right = "RE")
    R A C < E          (hook at 2)
    E R A C <          (hook at 3: end of word)

Usage
-----
    g = GADDAG()
    g.build_from_wordlist(["CARE", "CARD", ...])
    g.contains("CARE")   # True
    node = g.root        # traverse manually for move generation
"""

from __future__ import annotations

SEP = "<"   # separator character (must not be a letter)
EOW = ">"   # end-of-word marker


class GNode:
    """One node in the GADDAG trie."""

    __slots__ = ("children", "is_end")

    def __init__(self) -> None:
        self.children: dict[str, "GNode"] = {}
        self.is_end: bool = False  # True if a complete word ends here

    def child(self, ch: str) -> "GNode | None":
        return self.children.get(ch)

    def get_or_create(self, ch: str) -> "GNode":
        if ch not in self.children:
            self.children[ch] = GNode()
        return self.children[ch]


class GADDAG:
    def __init__(self) -> None:
        self.root = GNode()

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def insert(self, word: str) -> None:
        """Insert one word into the GADDAG."""
        word = word.upper()
        n = len(word)

        for hook in range(n):
            # Build the string: reversed prefix + SEP + suffix
            # reversed prefix: word[hook], word[hook-1], ..., word[0]
            # suffix        : word[hook+1], ..., word[n-1]
            node = self.root
            # Reversed prefix (including hook letter)
            for i in range(hook, -1, -1):
                node = node.get_or_create(word[i])
            # Separator
            node = node.get_or_create(SEP)
            # Suffix
            for i in range(hook + 1, n):
                node = node.get_or_create(word[i])
            node.is_end = True

    def build_from_wordlist(self, words: list[str]) -> None:
        for w in words:
            if w.isalpha():
                self.insert(w)

    def build_from_file(self, path: str) -> None:
        """Load one word per line from a plain-text dictionary file."""
        with open(path, encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word.isalpha():
                    self.insert(word)

    # ------------------------------------------------------------------
    # Simple lookup (for testing / validation)
    # ------------------------------------------------------------------

    def contains(self, word: str) -> bool:
        """Check if a word is in the dictionary (used for validation, not generation)."""
        word = word.upper()
        if not word:
            return False
        # Use the hook-at-0 path: word[0] SEP word[1] ... word[n-1]
        node = self.root
        node = node.child(word[0])
        if node is None:
            return False
        node = node.child(SEP)
        if node is None:
            return False
        for ch in word[1:]:
            node = node.child(ch)
            if node is None:
                return False
        return node.is_end

    # ------------------------------------------------------------------
    # Serialization — build once, reload in ~1 second
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Pickle the GADDAG to disk."""
        import pickle, pathlib
        pathlib.Path(path).write_bytes(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))

    @staticmethod
    def load(path: str) -> "GADDAG":
        """Load a previously saved GADDAG from disk."""
        import pickle, pathlib
        return pickle.loads(pathlib.Path(path).read_bytes())

    @staticmethod
    def load_or_build(dict_path: str, cache_path: str = "gaddag.pkl") -> "GADDAG":
        """
        Load from cache if it exists and is newer than the dictionary file,
        otherwise build from the text file and save the cache.
        """
        import pathlib
        cache = pathlib.Path(cache_path)
        source = pathlib.Path(dict_path)
        if cache.exists() and cache.stat().st_mtime >= source.stat().st_mtime:
            print(f"Loading GADDAG from cache ({cache})...")
            return GADDAG.load(cache_path)
        print(f"Building GADDAG from {dict_path} (this takes ~15s, cached after)...")
        g = GADDAG()
        g.build_from_file(dict_path)
        g.save(cache_path)
        print(f"Saved to {cache}.")
        return g

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def node_count(self) -> int:
        """Count total nodes (for debugging / memory analysis)."""
        seen: set[int] = set()
        stack = [self.root]
        while stack:
            n = stack.pop()
            if id(n) in seen:
                continue
            seen.add(id(n))
            stack.extend(n.children.values())
        return len(seen)
