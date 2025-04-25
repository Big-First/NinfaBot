# core/transformer_model.py

import torch
import torch.nn as nn # Importar nn
# Importar outras classes que TransformerModel usa
from .transformer_block import TransformerBlock
# from .feed_forward import FeedForward # Importado via TransformerBlock
# from .multi_head_self_attention import MultiHeadSelfAttention # Importado via TransformerBlock
from typing import List, Tuple # Importar tipos
import os # Para lidar com caminhos de arquivo e diretórios
# import math # Importado via MultiHeadSelfAttention
# import sys # Importado via MultiHeadSelfAttention

class TransformerModel(nn.Module):
    """
    Modelo completo Transformer (simplificado).
    """
    def __init__(self, vocab_size: int, max_seq_len: int = 512,
                 embedding_dim: int = 256, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        # --- PASSO 1: CHAMAR O CONSTRUTOR DA CLASSE BASE (SEMPRE O PRIMEIRO) ---
        super().__init__() # ESSENCIAL e a PRIMEIRA linha!

        # Determina o dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Modelo será executado em: {self.device}") # Manter print de device


        # Salvar parâmetros principais como atributos
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout # Salvar a probabilidade de dropout


        # --- PASSO 2: INSTANCIAR SUB-MÓDULOS E DEFINIR ATRIBUTOS ---
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim) # Embedding posicional aprendível

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, max_seq_len, dropout) # Passar todos os args
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embedding_dim) # Normalização final
        self.output_linear = nn.Linear(embedding_dim, vocab_size) # Camada de saída


        # --- PASSO 3: APLICAR INICIALIZAÇÕES OU CONFIGURAÇÕES GLOBAIS ---
        self.apply(self._init_weights) # Aplicar inicialização de pesos

        self.to(self.device) # Mover o modelo completo para o dispositivo


    # --- DEFINIR O MÉTODO _init_weights NA CLASSE ---
    def _init_weights(self, module):
       """
       Inicializa os pesos de módulos específicos.
       Aplicado recursivamente a todos os sub-módulos.
       """
       if isinstance(module, nn.Linear):
           torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
           if module.bias is not None:
               torch.nn.init.zeros_(module.bias)
       elif isinstance(module, nn.Embedding):
           torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
       elif isinstance(module, nn.LayerNorm):
           if hasattr(module, 'bias') and module.bias is not None:
               torch.nn.init.zeros_(module.bias)
           if hasattr(module, 'weight') and module.weight is not None:
                torch.nn.init.ones_(module.weight)

    # --- DEFINIR O MÉTODO forward ---
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids shape: (B, T_input) - Batch, Input Sequence Length
        # --- ESSENCIAIS PRIMEIRO ---
        B, T_input = input_ids.shape
        # ---------------------------

        # O forward do modelo espera uma sequência de tamanho ATÉ max_seq_len.
        # Se a sequência de entrada for maior, precisamos TRUNCAR aqui.
        if T_input > self.max_seq_len:
            # Pega APENAS os últimos max_seq_len tokens
            input_ids = input_ids[:, -self.max_seq_len:]
            T_input = self.max_seq_len # Atualiza T_input


        # Criar tensor de posições dinamicamente NO MESMO DISPOSITIVO DO INPUT
        positions = torch.arange(0, T_input, dtype=torch.long, device=input_ids.device) # (T_input,)
        positions = positions.unsqueeze(0).expand(B, T_input) # (B, T_input)


        # Calcular embeddings + posições
        token_embeds = self.token_embedding(input_ids) # [B, T_input, embed_dim]
        pos_embeds = self.position_embedding(positions) # [B, T_input, embed_dim]
        x = token_embeds + pos_embeds # [B, T_input, embed_dim]

        # Aplicar dropout no embedding (opcional)
        # x = torch.nn.functional.dropout(x, p=self.dropout_prob, training=self.training)

        # Passar pelos blocos Transformer
        for block in self.transformer_blocks: # Itera sobre os módulos em ModuleList
            x = block(x) # Chamada forward implícita

        # Normalização final
        x = self.final_norm(x)

        # Camada linear final para obter logits sobre o vocabulário
        logits = self.output_linear(x) # [B, T_input, vocab_size]

        return logits # Retornar o tensor de logits

    # --- DEFINIR PROPRIEDADES ---
    @property
    def MaxSeqLen(self) -> int:
        return self.max_seq_len

    @property
    def VocabSize(self) -> int:
        return self.vocab_size


    # --- DEFINIR MÉTODOS save/load ---
    def save(self, path: str):
        """
        Salva o estado do modelo (pesos) em um arquivo.
        """
        try:
            save_dir = os.path.dirname(path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                print(f"Diretório criado para salvar modelo: {save_dir}")

            torch.save(self.state_dict(), path)
            print(f"Estado do modelo salvo em: {path}")
        except Exception as e:
            print(f"ERRO ao salvar o modelo em {path}: {e}")
            # import traceback; traceback.print_exc()


    def load(self, path: str):
        """
        Carrega o estado do modelo (pesos) de um arquivo.
        """
        if not os.path.exists(path):
             # print(f"AVISO: Arquivo de modelo não encontrado em {path}. Não foi possível carregar.") # Este print já está na lógica condicional em app.py
             return # Sai da função se o arquivo não existir

        try:
            target_device = self.device
            state_dict = torch.load(path, map_location=target_device)
            self.load_state_dict(state_dict, strict=False)

            print(f"Estado do modelo carregado de {path}")
        except Exception as e:
             print(f"ERRO geral ao carregar modelo de {path}: {e}")
             # import traceback; traceback.print_exc()
             # raise e # Opcional: re-lançar se o carregamento for crítico