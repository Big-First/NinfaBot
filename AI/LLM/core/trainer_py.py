# core/trainer_py.py

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List # Importar List
from core.transformer_model import TransformerModel # Importar TransformerModel
from core.tokenizer_py import Tokenizer # Importar Tokenizer
from core.trainer_options import TrainerOptions # Importar TrainerOptions
from core.data_models import TrainingExample # Importar TrainingExample
from core.padding_helper import PaddingHelper # Importar PaddingHelper
import random # Para embaralhar o dataset
import os # Para criar diretório

class Trainer:
    """
    Classe responsável por treinar o modelo Transformer.
    """
    def __init__(self, model: TransformerModel, tokenizer: Tokenizer, options: TrainerOptions):
        super().__init__() # Trainer não herda de nn.Module, este super() NÃO é necessário. Remova se houver.
        self.model = model
        self.tokenizer = tokenizer
        self.options = options
        self.device = model.device # Usamos o device que já foi definido no modelo
        self.pad_token_id = tokenizer.GetPadTokenId()

        # Função de Loss: CrossEntropyLoss para predição de próximo token
        # ignore_index faz com que a loss seja 0 para os tokens de padding no target
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)

        # Otimizador: Adam (ou AdamW)
        self.optimizer = optim.Adam(self.model.parameters(), lr=options.learning_rate)

        print(f"Treinamento será executado em: {self.device}")
        if self.pad_token_id != -1:
             print(f"  CrossEntropyLoss ignorará o token ID: {self.pad_token_id}")
        else:
             print("  AVISO: PAD Token ID não encontrado ou inválido. CrossEntropyLoss pode não ignorar padding corretamente.")

        # Inicializar a flag de parada
        self._stop_training = False


    def train(self, dataset: List[TrainingExample]):
        if not dataset:
            print("Dataset de treinamento está vazio.")
            return

        print(f"🔧 Iniciando treinamento com {len(dataset)} exemplos...")
        print(f"  Épocas: {self.options.epochs}, Tamanho do Lote: {self.options.batch_size}, Seq Len: {self.options.max_seq_len}, LR: {self.options.learning_rate}")

        self.model.train() # Colocar modelo em modo de treinamento (ativa dropout, batch norm etc.)

        # Reinicializar a flag de parada para cada run de treino
        self._stop_training = False

        for epoch in range(1, self.options.epochs + 1):
            total_loss = 0
            batch_count = 0

            # Embaralhar o dataset a cada época (modifica 'dataset' no local)
            random.shuffle(dataset)

            # Iterar sobre o dataset EMBARALHADO
            # len(dataset) está correto agora
            for i in range(0, len(dataset), self.options.batch_size):
                if self._stop_training: # Verificar a flag no início do loop de batches
                     break # Sair do loop de batches imediatamente

                batch_items = dataset[i : i + self.options.batch_size] # Usar slicing na lista 'dataset'
                if not batch_items: continue

                input_sequences = []
                target_sequences = []

                for example in batch_items:
                    # Codificar input e output (sem a string literal <|endoftext|> nos outputs)
                    input_tokens = self.tokenizer.Encode(example.input, allow_special_tokens_in_text=False)
                    output_tokens = self.tokenizer.Encode(example.output, allow_special_tokens_in_text=False)

                    # Sequência completa para o modelo: [input] + [EOS (separador)] + [output] + [EOS (fim resposta)]
                    full_sequence = input_tokens + [self.tokenizer.GetEosTokenId()] + output_tokens + [self.tokenizer.GetEosTokenId()]

                    # Truncar a sequência completa
                    full_sequence = full_sequence[:self.options.max_seq_len]

                    model_input = full_sequence
                    # Target: Sequência shiftada para a esquerda + PAD no final
                    target_loss = full_sequence[1:] + [self.pad_token_id]

                    input_sequences.append(model_input)
                    target_sequences.append(target_loss)

                # Fazer Padding para o comprimento máximo do trainer
                padded_input_sequences_list = [
                    PaddingHelper.pad_sequence(seq, self.options.max_seq_len, self.pad_token_id)
                    for seq in input_sequences
                ]
                padded_target_sequences_list = [
                     PaddingHelper.pad_sequence(seq, self.options.max_seq_len, self.pad_token_id)
                     for seq in target_sequences
                ]

                # Converter listas de listas para tensores PyTorch [B, S]
                current_batch_size = len(padded_input_sequences_list)
                if current_batch_size == 0: continue
                # current_seq_len = len(padded_input_sequences_list[0]) # Não usado diretamente aqui

                input_batch_tensor = torch.tensor(padded_input_sequences_list, dtype=torch.long, device=self.device)
                target_batch_tensor = torch.tensor(padded_target_sequences_list, dtype=torch.long, device=self.device)

                # Etapa de Treinamento
                self.optimizer.zero_grad() # Limpar gradientes

                logits = self.model(input_batch_tensor) # Forward pass [B, S, V]

                # Calcular Loss (redimensionando para [B*S, V] e [B*S])
                loss = self.loss_fn(logits.view(-1, self.model.VocabSize), target_batch_tensor.view(-1))

                # Verificar NaN/Inf no TENSOR loss (corrigido)
                if torch.isnan(loss).item() or torch.isinf(loss).item():
                    print(f"AVISO: Loss NaN/Inf na época {epoch}, lote {batch_count}. Parando treinamento.")
                    self._stop_training = True # Set the flag
                    break # <-- Exit the batch loop

                loss.backward() # Calcular gradientes
                self.optimizer.step() # Atualizar pesos

                total_loss += loss.item() # Accumulate the float value
                batch_count += 1

            # Fim do loop de lotes (for i in range...)

            # Verificar a flag _stop_training APÓS O LOOP DE BATCHES para sair do loop de épocas
            if self._stop_training:
                 break # <-- Exit the epoch loop

            # Se o loop de batches completou SEM a flag ser setada
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            print(f"📚 Época {epoch}/{self.options.epochs} - Loss Média: {avg_loss:.4f}")

        # Fim do loop de épocas (for epoch in range...)

        # Lógica de salvamento/conclusão (verifica a flag _stop_training aqui)
        if self._stop_training:
             print("Treinamento interrompido devido a NaN/Inf Loss. Modelo NÃO salvo.")
        else:
            print("✅ Treinamento finalizado.")
            try:
                # Criar o diretório se não existir antes de salvar
                save_dir = os.path.dirname(self.options.save_path)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                    print(f"Diretório criado para salvar modelo: {save_dir}")

                self.model.save(self.options.save_path) # Chamar o método save do modelo
                print(f"💾 Modelo salvo em: {self.options.save_path}")
            except Exception as ex:
                print(f"❌ Erro ao salvar o modelo: {ex}")
                # import traceback; traceback.print_exc()