# core/padding_helper.py

import torch
from typing import List # Importar List

class PaddingHelper:
    @staticmethod
    def pad_sequence(tokens: List[int], length: int, pad_value: int) -> List[int]:
        """
        Adiciona padding a uma lista de tokens para atingir um comprimento específico.
        Trunca a lista se for maior que o comprimento desejado.
        """
        if len(tokens) > length:
            return tokens[:length]
        elif len(tokens) < length:
            return tokens + [pad_value] * (length - len(tokens))
        else:
            return tokens

    @staticmethod
    def pad_or_truncate_tensor(tensor: torch.Tensor, length: int, pad_value: int, device: torch.device) -> torch.Tensor:
        """
        Adiciona padding a um tensor 1D ou o trunca para atingir um comprimento específico.
        """
        if tensor.dim() != 1:
            raise ValueError("Tensor deve ser 1D.")
        current_len = tensor.shape[0]
        if current_len > length:
            truncated = tensor[:length].to(device)
            return truncated
        elif current_len < length:
            padding_tensor = torch.full((length - current_len,), pad_value,
                                        dtype=tensor.dtype, device=device)
            padded = torch.cat([tensor.to(device), padding_tensor], dim=0)
            return padded
        else:
            return tensor.to(device)