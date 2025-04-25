# Equivalent to TrainerOptions.cs

# Using dataclasses for simple data structures is Pythonic (Python 3.7+)
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TrainerOptions:
    """
    Opções de configuração para o treinamento do modelo Transformer.
    """
    batch_size: int = 8 # Valor padrão como no C# Startup
    max_seq_len: int = 64 # Valor padrão como no C# Startup (deve ser <= modelo.max_seq_len)
    epochs: int = 10 # Valor padrão como no C# Startup
    learning_rate: float = 1e-4 # Valor padrão como no C# Startup
    # Usamos field(default=None) e depois validamos/inicializamos no __post_init__ se necessário
    save_path: Optional[str] = None

    # O método __post_init__ é chamado após o __init__ gerado pelo dataclass
    def __post_init__(self):
        # Equivalente à validação de savePath no construtor C#
        if self.save_path is None:
             # Usamos ValueError em vez de ArgumentNullException em Python
             raise ValueError("save_path deve ser especificado.")
        # Validar que max_seq_len não excede um valor razoável (opcional, pode ser validado no Trainer)
        if self.max_seq_len <= 0:
             raise ValueError("max_seq_len deve ser positivo.")

# Exemplo de uso:
# options = TrainerOptions(save_path="model/my_model.pt")