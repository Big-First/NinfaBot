# core/feed_forward.py

import torch
import torch.nn as nn # Importar nn

class FeedForward(nn.Module):
    """
    Rede Feed-Forward usada em um bloco Transformer.
    """
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__() # Primeira linha!

        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ff_output = self.net(x)
        return ff_output # Retornar tensor