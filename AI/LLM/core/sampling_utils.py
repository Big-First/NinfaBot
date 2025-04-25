# core/sampling_utils.py

import torch
import torch.nn.functional as F # Usar F como alias é comum
import math # Importar math
from typing import List # Importar List
import sys # Para float('-inf')


class SamplingUtils:
    """
    Utilitários para amostragem de tokens a partir dos logits.
    """
    @staticmethod
    def sample_next_token(logits: torch.Tensor, temperature: float = 0.7, top_k: int = 0, top_p: float = 0.0) -> int:
        """
        Amostra um token a partir dos logits usando temperatura, top-k ou top-p.

        Args:
            logits: Logits do último token (shape [vocab_size])
            temperature: Controla a aleatoriedade (valores > 0). 0 para greedy.
            top_k: Considera apenas os K tokens mais prováveis (0 para desativar).
            top_p: Considera o menor conjunto de tokens cuja probabilidade cumulativa é >= P (0 para desativar).
        Returns:
            ID do token amostrado.
        """
        if logits.dim() != 1:
            raise ValueError(f"Logits devem ser 1D (vetor de vocabulário), mas tem shape {logits.shape}")

        temp_logits = logits.clone()

        # 1. Greedy Sampling (se temperatura for muito baixa)
        if temperature <= 1e-6:
            result = temp_logits.argmax().item()
            return result

        # 2. Aplicar Temperatura
        tempered_logits = temp_logits / temperature

        # 3. Calcular Probabilidades usando Softmax
        probs = F.softmax(tempered_logits, dim=0) # Usar F.softmax

        # 4. Aplicar Top-K ou Top-P
        filtered_probs: torch.Tensor
        if top_k > 0:
            filtered_probs = SamplingUtils._apply_top_k(probs, top_k)
        elif 0.0 < top_p < 1.0:
            filtered_probs = SamplingUtils._apply_top_p(probs, top_p)
        else:
            filtered_probs = probs.clone() # Clone para garantir que filtered_probs é um novo tensor

        # 5. Amostrar da distribuição resultante
        # Verificar soma das probabilidades filtras antes de multinomial
        if filtered_probs.sum().item() < 1e-9:
             # print("AVISO: Probabilidades filtras somam ~0. Fallback para Greedy.") # Opcional print
             result = temp_logits.argmax().item() # Fallback para greedy usando os logits originais (temperados)
             return result

        # Verificar NaN/Inf após filtragem
        if torch.isnan(filtered_probs).any().item() or torch.isinf(filtered_probs).any().item(): # Usar .item() para obter bool
             # print("AVISO: Probabilidades filtras contêm NaN/Inf. Fallback para Greedy.") # Opcional print
             result = temp_logits.argmax().item()
             return result

        # Amostragem multinomial
        next_token_tensor = torch.multinomial(filtered_probs, num_samples=1)

        # Extrair o valor int
        next_token = next_token_tensor.item()

        return next_token

    @staticmethod
    def _apply_top_k(probs: torch.Tensor, k: int) -> torch.Tensor:
        """
        Filtra probabilidades para manter apenas os K maiores (Top-K).
        Zera as outras probabilidades e re-normaliza. Assume probs é 1D.
        """
        k = max(1, min(k, probs.shape[0]))

        top_k_result = torch.topk(probs, k, dim=0)
        top_values = top_k_result.values
        top_indices = top_k_result.indices

        filtered_probs = torch.full_like(probs, 0.0)
        filtered_probs.index_put_((top_indices,), top_values) # index_put_ espera tupla(tensor de índices)

        sum_probs = filtered_probs.sum()
        if sum_probs.item() > 1e-9:
            filtered_probs = filtered_probs / sum_probs

        return filtered_probs

    @staticmethod
    def _apply_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
        """
        Filtra probabilidades mantendo o menor conjunto cuja soma cumulativa >= p (Top-P).
        Zera as outras probabilidades e re-normaliza. Assume probs é 1D.
        """
        p = max(0.0, min(p, 1.0))
        if p <= 1e-6:
             return SamplingUtils._apply_top_k(probs, 1) # Fallback para TopK=1 (greedy)
        if p >= 1.0 - 1e-6: # Usar uma pequena margem devido a ponto flutuante
             return probs.clone() # Retorna uma cópia

        sorted_result = torch.sort(probs, dim=0, descending=True)
        sorted_probs = sorted_result.values
        sorted_indices = sorted_result.indices

        cumulative_probs = torch.cumsum(sorted_probs, dim=0)

        # Encontrar os índices a manter (onde cumulative_probs <= p)
        sorted_indices_to_keep_mask = cumulative_probs <= p
        # Garantir que pelo menos o token mais provável é mantido
        sorted_indices_to_keep_mask[0] = True

        # Selecionar os valores e índices originais a manter
        original_indices_to_keep = sorted_indices.masked_select(sorted_indices_to_keep_mask)
        values_to_keep = sorted_probs.masked_select(sorted_indices_to_keep_mask)

        filtered_probs = torch.full_like(probs, 0.0)
        if values_to_keep.shape[0] > 0: # Evitar index_put_ com tensor vazio
            filtered_probs.index_put_((original_indices_to_keep,), values_to_keep) # Usar tupla de índice

        sum_probs = filtered_probs.sum()
        if sum_probs.item() > 1e-9:
             filtered_probs = filtered_probs / sum_probs

        return filtered_probs