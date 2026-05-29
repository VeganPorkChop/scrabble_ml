"""
ScrabbleNet — neural network that estimates the value of a Scrabble move.

Architecture: three-stream "late fusion" network
─────────────────────────────────────────────────

  Board stream (CNN)
  ┌─────────────────────────────────────────────┐
  │ Input: [27 x 15 x 15]  (letter channels)    │
  │  Conv2d(27→64, 3x3)  + BatchNorm + ReLU     │
  │  Conv2d(64→64, 3x3)  + BatchNorm + ReLU     │
  │  Conv2d(64→32, 3x3)  + BatchNorm + ReLU     │
  │  GlobalAvgPool  →  [32]                      │
  └──────────────────── 32 units ───────────────┘

  Rack stream (MLP)
  ┌─────────────────────────────────────────────┐
  │ Input: [27]  (normalised tile counts)        │
  │  Linear(27→32)  + ReLU                      │
  └──────────────────── 32 units ───────────────┘

  Move stream (MLP)
  ┌─────────────────────────────────────────────┐
  │ Input: [N_MOVE_FEATURES]  (14 scalars)      │
  │  Linear(14→32)  + ReLU                      │
  └──────────────────── 32 units ───────────────┘

  Fusion head
  ┌─────────────────────────────────────────────┐
  │ Concat [32+32+32] = 96 units                │
  │  Linear(96→128) + ReLU + Dropout(0.2)       │
  │  Linear(128→64) + ReLU                      │
  │  Linear(64→1)                               │
  │  Output: scalar — estimated score advantage │
  └─────────────────────────────────────────────┘

Why GlobalAvgPool instead of Flatten?
  Flattening a 32×15×15 feature map gives 7,200 inputs to the head —
  expensive and prone to overfitting.  Global average pooling compresses
  each spatial channel to its mean, producing 32 values while retaining
  "where each letter type is distributed on the board".

Why BatchNorm?
  Scrabble boards are sparse (most squares empty early game, dense late).
  BatchNorm normalises activations across the batch so the network doesn't
  over-weight the fact that most cells are 0.

Output interpretation:
  The model outputs a single scalar representing the expected score margin
  (our score − opponent's score) from the current position if we play this
  move.  During training we use the ACTUAL game outcome as the supervision
  signal (Monte Carlo return).
"""

import torch
import torch.nn as nn
from features import BOARD_CHANNELS, BOARD_SIZE, N_MOVE_FEATURES


class BoardEncoder(nn.Module):
    """Small CNN that processes the 15x15 board."""

    def __init__(self, out_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: 27 → 64 channels, 3×3 kernels, same-padding keeps 15×15
            nn.Conv2d(BOARD_CHANNELS, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Layer 2: 64 → 64
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Layer 3: 64 → 32 (compress channels before pooling)
            nn.Conv2d(64, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),

            # Global average pool: [B, 32, 15, 15] → [B, 32, 1, 1] → [B, 32]
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScrabbleNet(nn.Module):
    """
    Full three-stream move-value network.

    Inputs
    ------
    board  : FloatTensor [B, 27, 15, 15]
    rack   : FloatTensor [B, 27]
    move_f : FloatTensor [B, N_MOVE_FEATURES]

    Output
    ------
    value  : FloatTensor [B, 1]  — estimated score margin
    """

    STREAM_DIM = 32    # output size for each stream
    HIDDEN_DIM = 128

    def __init__(self) -> None:
        super().__init__()

        # ── Streams ────────────────────────────────────────────
        self.board_encoder = BoardEncoder(out_dim=self.STREAM_DIM)

        self.rack_encoder = nn.Sequential(
            nn.Linear(BOARD_CHANNELS, self.STREAM_DIM),
            nn.ReLU(),
        )

        self.move_encoder = nn.Sequential(
            nn.Linear(N_MOVE_FEATURES, self.STREAM_DIM),
            nn.ReLU(),
        )

        # ── Fusion head ────────────────────────────────────────
        fused_dim = self.STREAM_DIM * 3   # 96

        self.head = nn.Sequential(
            nn.Linear(fused_dim, self.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(p=0.2),            # regularisation

            nn.Linear(self.HIDDEN_DIM, 64),
            nn.ReLU(),

            nn.Linear(64, 1),             # scalar output
        )

        # Weight initialisation — Xavier uniform for linear layers
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(
        self,
        board: torch.Tensor,
        rack: torch.Tensor,
        move_f: torch.Tensor,
    ) -> torch.Tensor:
        b_feat = self.board_encoder(board)      # [B, 32]
        r_feat = self.rack_encoder(rack)        # [B, 32]
        m_feat = self.move_encoder(move_f)      # [B, 32]

        fused = torch.cat([b_feat, r_feat, m_feat], dim=1)   # [B, 96]
        return self.head(fused)                              # [B, 1]

    # ── Convenience: single-move inference ────────────────────

    def score_move(
        self,
        board_t: torch.Tensor,   # [27, 15, 15]
        rack_t: torch.Tensor,    # [27]
        move_t: torch.Tensor,    # [N_MOVE_FEATURES]
    ) -> float:
        """Return the estimated value as a Python float (no grad)."""
        self.eval()
        with torch.no_grad():
            val = self(
                board_t.unsqueeze(0),
                rack_t.unsqueeze(0),
                move_t.unsqueeze(0),
            )
        return val.item()

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
