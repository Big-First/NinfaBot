# core/multi_head_self_attention.py

import torch
import torch.nn as nn # Importar nn
import math # Importar math
from typing import Tuple # Importar Tuple
import sys # Para float('-inf')

class MultiHeadSelfAttention(nn.Module):
    """
    Mecanismo de Atenção Auto-Atencional Multi-Cabeça com máscara causal.
    """
    def __init__(self, embed_dim: int, num_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__() # Primeira linha!

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) deve ser divisível por num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.dropout_prob = dropout
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        # Pré-calcular a máscara causal
        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask) # Registrar como buffer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, D) - Batch, Sequence Length, Embedding Dimension
        B, T, D = x.shape

        # Obter a máscara registrada e fatiar para o comprimento atual da sequência
        causal_mask_sliced = self.causal_mask[:T, :T] # Shape (T, T)

        qkv_out = self.qkv(x) # Shape (B, T, D*3)

        # Dividir qkv em q, k, v e reorganizar para multi-cabeça
        q = qkv_out.chunk(3, dim=-1)[0].view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, T, head_dim)
        k = qkv_out.chunk(3, dim=-1)[1].view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, T, head_dim)
        v = qkv_out.chunk(3, dim=-1)[2].view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, T, head_dim)

        # Calcular scores de atenção (Q @ K^T) / sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) # (B, num_heads, T, T)

        # Aplicar a máscara causal (broadcastável)
        # A máscara (T, T) precisa ser expandida para (B, num_heads, T, T) para broadcast
        scores = scores.masked_fill(causal_mask_sliced.unsqueeze(0).unsqueeze(0), float('-inf')) # Usar float('-inf')

        # Calcular pesos de atenção e aplicar dropout
        attn_weights = torch.softmax(scores, dim=-1) # (B, num_heads, T, T)
        attn_output_weighted = self.dropout(attn_weights) # (B, num_heads, T, T)

        # Multiplicar pesos de atenção por V
        outTensor = torch.matmul(attn_output_weighted, v) # (B, num_heads, T, head_dim)

        # Concatenar as cabeças e projetar
        outTensor = outTensor.transpose(1, 2).contiguous().view(B, T, self.embed_dim) # (B, T, embed_dim)

        final_output = self.proj(outTensor) # (B, T, embed_dim)
        return final_output # Retornar tensor