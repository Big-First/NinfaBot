# Script principal para treinar ou rodar inferência com o modelo LLM

import torch
import os
from typing import List
# Importar as classes adaptadas
from core.tokenizer_py import Tokenizer
from core.transformer_model import TransformerModel
from core.trainer_py import tr
from core.trainer_options import TrainerOptions
from core.data_models import TrainingExample, PromptRequest
from core.sampling_utils import SamplingUtils # Importar SamplingUtils

# --- Configurações ---
MODEL_SAVE_PATH = os.path.join("model", "ninfa_py.pt") # Caminho para salvar/carregar o modelo
# Parâmetros do modelo (devem ser consistentes!)
VOCAB_SIZE = 50257 # Vocabulário do gpt2
MODEL_MAX_SEQ_LEN = 64 # max_seq_len que o modelo foi construído para lidar
EMBEDDING_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1

# Parâmetros do Trainer (devem ser consistentes com o modelo, especialmente max_seq_len)
TRAINER_BATCH_SIZE = 8
TRAINER_EPOCHS = 10
TRAINER_LEARNING_RATE = 1e-4
TRAINER_MAX_SEQ_LEN = MODEL_MAX_SEQ_LEN # Usar o mesmo do modelo

# --- Função para Configurar e Carregar/Criar Modelo ---
def setup_model(vocab_size: int, model_max_seq_len: int, embedding_dim: int,
                num_heads: int, num_layers: int, dropout: float, save_path: str) -> TransformerModel:
    """
    Cria ou carrega o modelo Transformer.
    """
    model = TransformerModel(vocab_size, model_max_seq_len, embedding_dim,
                             num_heads, num_layers, dropout)

    # Tentar carregar o modelo se ele existir
    if os.path.exists(save_path):
        try:
            model.load(save_path)
            print(f"Modelo carregado com sucesso de: {save_path}")
        except Exception as ex:
            print(f"ATENÇÃO: Erro ao carregar modelo de {save_path}. Iniciando com pesos aleatórios. Erro: {ex}")
            # Decida se quer travar ou continuar
            # raise
    else:
         print(f"Arquivo de modelo não encontrado em: {save_path}. Iniciando com pesos aleatórios.")
         # Garantir que o diretório "model" existe se for o primeiro treinamento
         save_dir = os.path.dirname(save_path)
         if save_dir and not os.path.exists(save_dir):
             os.makedirs(save_dir)
             print(f"Diretório criado: {save_dir}")


    # Mover modelo para o dispositivo correto (CPU ou GPU) - Já acontece no __init__ do modelo, mas é seguro chamar novamente
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Modelo final no dispositivo: {model.device}")

    # Colocar modelo em modo de avaliação por padrão (para inferência)
    model.eval()

    return model

# --- Função para Executar Treinamento ---
def run_training(model: TransformerModel, tokenizer: Tokenizer):
    """
    Prepara e executa o treinamento.
    """
    # Criar opções do trainer
    trainer_options = TrainerOptions(
        batch_size=TRAINER_BATCH_SIZE,
        max_seq_len=TRAINER_MAX_SEQ_LEN,
        epochs=TRAINER_EPOCHS,
        learning_rate=TRAINER_LEARNING_RATE,
        save_path=MODEL_SAVE_PATH
    )

    # Criar uma instância do Trainer
    trainer = Trainer(model, tokenizer, trainer_options)

    # Exemplo de dataset de treinamento (substitua por seus dados reais)
    # Isto é apenas um placeholder. Você precisa carregar seus dados de um arquivo/fonte.
    training_dataset = [
        TrainingExample(input="Qual a capital do Brasil?", output="Brasília."),
        TrainingExample(input="Quem descobriu o Brasil?", output="Pedro Álvares Cabral."),
        TrainingExample(input="O que é um transformador?", output="Um transformador é um tipo de modelo de rede neural."),
        TrainingExample(input="Olá, como você está?", output="Estou bem, obrigado!"),
        TrainingExample(input="Conte uma história curta.", output="Era uma vez, numa terra distante, uma pequena fada."),
        # Adicione mais exemplos aqui...
    ] * 10 # Multiplica para ter mais exemplos para o treino de demonstração

    # Executar o treinamento
    trainer.train(training_dataset)

# --- Função para Executar Inferência ---
def run_inference(model: TransformerModel, tokenizer: Tokenizer, prompt_request: PromptRequest) -> str:
    """
    Executa a geração de texto (inferência) com o modelo.
    """
    if not prompt_request.input:
        print("Texto de entrada (prompt) está vazio para inferência.")
        return ""

    model.eval() # Colocar o modelo em modo de avaliação (desativa dropout, etc.)

    # Não precisamos de `using var scope = torch.NewDisposeScope();` como no C# TorchSharp.
    # Python + PyTorch gerenciam a memória automaticamente via coletor de lixo.

    # Codificar o prompt de entrada
    # allow_special_tokens_in_text=False para tratar tokens especiais no prompt como texto comum
    input_tokens = tokenizer.Encode(prompt_request.input, allow_special_tokens_in_text=False)
    generated_tokens = list(input_tokens) # Começa a lista gerada com os tokens do prompt

    # Obter o ID do token EOS para parar a geração
    eos_token_id = tokenizer.GetEosTokenId()
    # Obter o max_seq_len do modelo
    model_max_seq_len = model.MaxSeqLen


    print(f"Prompt inicial: '{prompt_request.input}'")
    print(f"Generating up to {prompt_request.max_new_tokens} new tokens...")

    # Loop de geração de tokens
    for _ in range(prompt_request.max_new_tokens):
        # A sequência de entrada para o modelo na inferência é a sequência gerada ATÉ AGORA.
        # O modelo fará o truncamento interno se current_sequence_tensor.shape[1] > model_max_seq_len.
        # Precisamos passar apenas os *últimos* `model_max_seq_len` tokens como entrada para o modelo.

        # Pega os últimos `model_max_seq_len` tokens da sequência gerada
        current_sequence = generated_tokens[-model_max_seq_len:] # Python slicing pega do -model_max_seq_len até o fim

        # Converter a lista de tokens para um tensor PyTorch de shape [1, current_seq_len]
        # Adicionar uma dimensão de batch (tamanho 1)
        current_sequence_tensor = torch.tensor([current_sequence], dtype=torch.long, device=model.device) # shape [1, T_atual]

        # Forward pass. Retorna logits [B, T_atual_real, V]
        # O modelo já truncou o input para T_atual_real = min(T_atual, model_max_seq_len)
        # Estamos interessados apenas no logit do *último* token gerado, que prevê o PRÓXIMO token.
        # Portanto, pegamos a saída correspondente ao último token da sequência de *entrada* (que pode ser truncada).
        # A saída tem o mesmo comprimento que a entrada efetiva.
        # O último token da entrada efetiva é o índice -1 (ou shape[1]-1).
        with torch.no_grad(): # Não calcular gradientes durante a inferência
             output = model(current_sequence_tensor) # shape [1, T_atual_real, vocab_size]

        # Obter os logits para o *próximo* token (correspondente ao último token da entrada efetiva)
        logits = output[0, -1, :] # Pega o item 0 do batch, o último token (-1), e todos os logits (:)
                                  # shape [vocab_size]

        # Amostrar o próximo token usando SamplingUtils
        next_token = SamplingUtils.sample_next_token(
            logits,
            prompt_request.temperature,
            prompt_request.top_k,
            prompt_request.top_p
        )

        # Os tensores current_sequence_tensor, output, logits sairão do escopo
        # e serão elegíveis para coleta de lixo.

        # Se o token gerado for EOS, parar
        if next_token == eos_token_id:
            print("Generated EOS token, stopping generation.")
            break

        # Adicionar o token gerado à sequência
        generated_tokens.append(next_token)

        # Opcional: Limite total de tokens gerados (input original + novos)
        # if len(generated_tokens) >= MODEL_MAX_SEQ_LEN * 2: # Exemplo de limite total
        #      print("Reached total token limit, stopping generation.")
        #      break


    # Decodificar a sequência gerada (desde o início, incluindo o prompt original)
    decoded_text = tokenizer.Decode(generated_tokens)

    # O C# endpoint retornava APENAS o texto gerado. Aqui retornamos o texto completo
    # (prompt + resposta). Você pode ajustar se quiser retornar apenas a resposta.
    # Para retornar apenas a resposta: encontrar onde o prompt termina na lista generated_tokens
    # e decodificar a partir daí.
    # Find where the original prompt ended:
    # prompt_end_index = len(input_tokens)
    # decoded_text = tokenizer.Decode(generated_tokens[prompt_end_index:]) # Decode only the generated part

    return decoded_text

# --- Ponto de Entrada Principal ---
if __name__ == "__main__":
    # Setup o tokenizer
    tokenizer = Tokenizer("gpt2")

    # Setup o modelo (cria ou carrega)
    model = setup_model(VOCAB_SIZE, MODEL_MAX_SEQ_LEN, EMBEDDING_DIM,
                        NUM_HEADS, NUM_LAYERS, DROPOUT, MODEL_SAVE_PATH)

    # --- Escolha: Rodar Treinamento ou Inferência ---

    # Para Rodar Treinamento:
    # print("\n--- RODANDO TREINAMENTO ---")
    # run_training(model, tokenizer)
    # # Após treinar, o modelo é salvo automaticamente.
    # # Você pode rodar inferência depois carregando este modelo salvo.


    # Para Rodar Inferência:
    print("\n--- RODANDO INFERÊNCIA ---")
    # Certifique-se de que um modelo treinado existe no MODEL_SAVE_PATH
    if not os.path.exists(MODEL_SAVE_PATH):
         print(f"Não é possível rodar inferência: Arquivo de modelo '{MODEL_SAVE_PATH}' não encontrado.")
         print("Por favor, rode o treinamento primeiro ou forneça um modelo pré-treinado.")
    else:
        # Certifique-se de que o modelo carregado está em modo eval()
        model.eval() # Já é o padrão após setup_model, mas reconfirmar

        # Criar uma solicitação de inferência
        inference_request = PromptRequest(
            input="O que é inteligência artificial?",
            max_new_tokens=100, # Quantos tokens novos gerar no máximo
            temperature=0.8,
            top_k=50, # Considerar os 50 mais prováveis
            top_p=0.95 # Ou usar top-p (mutuamente exclusivo com top-k efetivamente se ambos > 0)
        )

        # Rodar a inferência
        generated_text = run_inference(model, tokenizer, inference_request)

        print("\n--- RESULTADO DA INFERÊNCIA ---")
        print(generated_text)
        print("-------------------------------")