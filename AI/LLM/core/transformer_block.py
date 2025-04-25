# core/transformer_block.py

import torch
import torch.nn as nn # Importar nn
from .multi_head_self_attention import MultiHeadSelfAttention # Importar MultiHeadSelfAttention
from .feed_forward import FeedForward # Importar FeedForward
from typing import List, Tuple # Importar tipos

class TransformerBlock(nn.Module):
    """
    Um único bloco Transformer (atenção + feed-forward com conexões residuais e normalização).
    """
    def __init__(self, embed_dim: int, num_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__() # Primeira linha!

        self.norm1 = nn.LayerNorm(embed_dim)
        # Passar max_seq_len e dropout para MultiHeadSelfAttention
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, max_seq_len, dropout)
        self.dropout1 = nn.Dropout(dropout) # Dropout após a atenção

        self.norm2 = nn.LayerNorm(embed_dim)
        # Passar dropout para FeedForward
        self.ff = FeedForward(embed_dim, dropout)
        self.dropout2 = nn.Dropout(dropout) # Dropout após a feed-forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Normalization: LayerNorm -> Sublayer -> Add & Norm
        # Atenção com conexão residual
        residual1 = x # Capturar residual ANTES da normalização
        norm1_out = self.norm1(x)
        attn_output = self.attn(norm1_out)
        dropout1_out = self.dropout1(attn_output) # Aplicar dropout na saída da sublayer
        x = residual1 + dropout1_out # Adicionar residual (Adicionar ANTES da próxima normalização)

        # Feed-Forward com conexão residual
        residual2 = x # Capturar residual ANTES da normalização
        norm2_out = self.norm2(x)
        ff_output = self.ff(norm2_out)
        dropout2_out = self.dropout2(ff_output) # Aplicar dropout na saída da sublayer
        x = residual2 + dropout2_out # Adicionar residual

        return x # Retornar tensor