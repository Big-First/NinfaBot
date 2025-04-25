# core/data_models.py

from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class TrainingExample:
    input: str
    output: str

@dataclass
class PromptRequest:
    input: str
    max_new_tokens: int = 50
    temperature: float = 0.7
    top_k: int = 0
    top_p: float = 0.9