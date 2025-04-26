import torch
import os
import json
from flask import Flask, request, jsonify # Importar no topo
from flask_socketio import SocketIO, emit # Importar no topo
from typing import List, Optional # Importar Optional para type hinting

# Importar as classes adaptadas do seu pacote core
from core.tokenizer_py import Tokenizer
from core.transformer_model import TransformerModel
from core.trainer_py import Trainer
from core.trainer_options import TrainerOptions
from core.data_models import PromptRequest # Para tipagem
# SamplingUtils e TrainingExample serão importados localmente onde usados ou no topo
from core.sampling_utils import SamplingUtils # <--- Importar SamplingUtils de core
# from core.data_models import TrainingExample # Importado no topo ou localmente no treino

# Importar a função que carrega o dataset
from core.data_utils import get_training_dataset


# --- Configurações ---
# Usar um caminho absoluto para o arquivo do modelo
# os.path.dirname(__file__) obtém o diretório onde app.py está (H:/GitHub/NinfaBot/AI/LLM/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "model", "ninfa_py.pt")

VOCAB_SIZE = 50257          # Vocabulário do gpt2 (tiktoken)
MODEL_MAX_SEQ_LEN = 64      # max_seq_len para o qual o modelo foi construído
EMBEDDING_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1

# Parâmetros do Trainer (consistentes com o modelo)
TRAINER_BATCH_SIZE = 8
TRAINER_EPOCHS = 10
TRAINER_LEARNING_RATE = 1e-4
TRAINER_MAX_SEQ_LEN = MODEL_MAX_SEQ_LEN # Usar o mesmo do modelo


# --- Inicialização do Flask App e SocketIO (Globais) ---
# Crie as instâncias globalmente AQUI no topo do módulo
app = Flask(__name__)
# Você pode configurar a chave secreta para sessões em produção
# app.config['SECRET_KEY'] = 'sua_chave_muito_secreta_e_aleatoria_aqui'

# Configurações SocketIO (modo assíncrono, logs)
# async_mode='gevent' é recomendado para performance com gevent
socketio = SocketIO(app, async_mode='gevent', logger=True, engineio_logger=True)


# --- Instâncias Globais para o Modelo e Tokenizer ---
# Serão inicializadas na função de setup que roda no __main__
tokenizer: Optional[Tokenizer] = None
model: Optional[TransformerModel] = None


# --- Função de Setup da Aplicação (Contém a lógica de inicialização do Modelo/Tokenizer/Treino) ---
def setup_application():
    """
    Configura o Tokenizer, Modelo e lida com o carregamento/treinamento.
    Esta função é chamada UMA VEZ no início do script.
    """
    global tokenizer, model # Indica que estamos usando as variáveis globais

    print("🚀 Iniciando aplicação Flask...") # Esta linha agora estará DENTRO da função de setup

    # --- Setup do Tokenizer ---
    print("⏳ Configurando Tokenizer...")
    try:
        tokenizer = Tokenizer("gpt2") # Inicializa o tokenizer
        print("✅ Tokenizer configurado.")
    except Exception as e:
        print(f"❌ ERRO FATAL ao configurar Tokenizer: {e}")
        # import traceback; traceback.print_exc() # Descomente para log detalhado
        tokenizer = None # Define como None se falhar


    # --- Setup e Carregamento/Treinamento do Modelo ---
    # Só tenta configurar o modelo se o tokenizer foi configurado com sucesso
    if tokenizer:
        print("⏳ Configurando e carregando/criando Modelo Transformer...")
        try:
            # Cria uma nova instância do modelo (com pesos aleatórios inicialmente)
            model = TransformerModel(
                vocab_size=VOCAB_SIZE,
                max_seq_len=MODEL_MAX_SEQ_LEN,
                embedding_dim=EMBEDDING_DIM,
                num_heads=NUM_HEADS,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT
            )

            # --- Lógica Explícita: Carregar SE EXISTE, Treinar SE NÃO EXISTE ---
            if os.path.exists(MODEL_SAVE_PATH):
                 print(f"✅ Arquivo de modelo encontrado em: {MODEL_SAVE_PATH}. Tentando carregar...")
                 # model.load() retorna True/False agora
                 load_success = model.load(MODEL_SAVE_PATH)

                 if load_success:
                     print(f"✅ Modelo carregado com sucesso de: {MODEL_SAVE_PATH}.")
                 else:
                     # Se o arquivo existia mas o carregamento falhou (corrompido, incompatível, etc.)
                     print(f"❌ FALHA ao carregar modelo de {MODEL_SAVE_PATH}. Iniciando TREINAMENTO como fallback.")
                     # Prossiga para o bloco de treinamento abaixo
                     train_needed = True # Força o treino
            else:
                 # Arquivo NÃO encontrado, iniciar treinamento
                 print(f"⚠️ Arquivo de modelo não encontrado em: {MODEL_SAVE_PATH}.")
                 print("⚠️ Iniciando TREINAMENTO (isso pode levar tempo! Servidor bloqueado).")
                 train_needed = True # Indica que o treino é necessário


            if train_needed:
                 # Garantir que o diretório "model" existe antes de treinar e salvar
                 save_dir = os.path.dirname(MODEL_SAVE_PATH)
                 if save_dir and not os.path.exists(save_dir):
                     os.makedirs(save_dir, exist_ok=True)
                     print(f"Diretório criado para salvar modelo: {save_dir}")

                 # --- INICIAR TREINAMENTO AQUI ---
                 try:
                     # Importar classes e funções de treino (algumas já no topo)
                     # from core.trainer_py import Trainer
                     # from core.trainer_options import TrainerOptions
                     from core.data_models import TrainingExample # Importar TrainingExample aqui para garantir

                     trainer_options = TrainerOptions(
                         batch_size=TRAINER_BATCH_SIZE,
                         max_seq_len=TRAINER_MAX_SEQ_LEN,
                         epochs=TRAINER_EPOCHS,
                         learning_rate=TRAINER_LEARNING_RATE,
                         save_path=MODEL_SAVE_PATH
                     )
                     trainer = Trainer(model, tokenizer, trainer_options) # Cria o trainer

                     training_dataset = get_training_dataset() # Carrega os dados de treino
                     training_dataset = training_dataset * 5 # Opcional: Multiplica

                     print(f"--- DEBUG: Dataset de treino carregado com {len(training_dataset)} exemplos. ---")

                     print("DEBUG - Chamando trainer.train()...")
                     trainer.train(training_dataset) # <--- EXECUTA O TREINAMENTO COMPLETO
                     print("DEBUG - Chamada a trainer.train() completa.")

                     print("✅ Treinamento na inicialização finalizado. Modelo salvo.")

                 except Exception as train_ex:
                     print(f"❌ ERRO DURANTE O TREINAMENTO NA INICIALIZAÇÃO: {train_ex}")
                     # import traceback; traceback.print_exc() # Descomente para log detalhado
                     # Decida se quer travar o servidor ou continuar com modelo não treinado/parcialmente treinado
                     # raise train_ex # Descomente para travar o servidor se o treino falhar
            # --- Fim da Lógica de Treinamento ---


            # Colocar modelo em modo de avaliação por padrão (para inferência)
            # Este passo acontece SEMPRE APÓS tentar carregar OU treinar
            model.eval() # Mover para eval mode, pronto para inferência
            print("✅ Modelo configurado.")

        except Exception as e:
            print(f"❌ ERRO FATAL ao configurar Modelo Transformer: {e}")
            # import traceback; traceback.print_exc() # Descomente para log detalhado
            model = None # Define como None se falhar


# --- DEFINIÇÃO DAS ROTAS HTTP E EVENTOS WebSocket ---
# Estas funções acessam as variáveis globais `app` e `socketio` (criadas no topo)
# e `model` e `tokenizer` (inicializadas em setup_application)

@app.route('/Input', methods=['POST'])
def handle_input_http():
    # Acessa as variáveis globais `model` e `tokenizer`
    global model, tokenizer
    # request e jsonify são importados no topo

    # Verificar se modelo e tokenizer estão disponíveis
    if model is None or tokenizer is None:
        return jsonify({"error": "Servidor não configurado corretamente (modelo/tokenizer indisponível)."}), 500

    # Obter dados da requisição JSON
    request_data = request.json
    if not request_data:
        return jsonify({"error": "Requisição deve ser JSON."}), 400

    # Extrair prompt e parâmetros de geração
    prompt_text = request_data.get('Input')
    if not prompt_text:
        return jsonify({"error": "'Input' não fornecido na requisição JSON."}), 400

    max_new_tokens = request_data.get('MaxNewTokens', 50)
    temperature = request_data.get('Temperature', 0.7)
    top_k = request_data.get('TopK', 0)
    top_p = request_data.get('TopP', 0.9)

    # --- Lógica de Geração (Inferência) ---
    try:
        # SamplingUtils é acessível aqui (importado no topo)
        print("Input: ", prompt_text)
        generated_text = generate_text_inference(prompt_text, max_new_tokens, temperature, top_k, top_p, model, tokenizer)
        print("Response: ", generated_text)
        return jsonify({"response": generated_text})
    except Exception as e:
        print(f"Erro durante a geração HTTP: {e}")
        # import traceback; traceback.print_exc() # Descomente para log detalhado
        return jsonify({"error": f"Erro durante a geração: {e}"}), 500


@socketio.on('connect')
def handle_connect():
    # emit é acessível (importado no topo)
    # Acessa as variáveis globais `model` e `tokenizer`
    global model, tokenizer
    print('➡️ Client connected via WebSocket!')
    if model is None or tokenizer is None:
        emit('error', {'message': 'Server is not ready (model or tokenizer unavailable)'})
        # from flask_socketio import disconnect # Importar localmente se usar
        # disconnect()


@socketio.on('disconnect')
def handle_disconnect():
    print('⬅️ Client disconnected.')


@socketio.on('generate_text')
def handle_generate_text(data):
    # emit é acessível (importado no topo)
    # Acessa as variáveis globais `model` e `tokenizer`
    global model, tokenizer
    # SamplingUtils é acessível (importado no topo)

    if model is None or tokenizer is None:
        emit('error', {'message': 'Server is not ready (model or tokenizer unavailable)'})
        return # Parar a execução deste handler se o servidor não estiver pronto

    print(f"Received generate_text event with data: {data}")

    prompt_text = data.get('Input')
    if not prompt_text:
        emit('error', {'message': "'Input' not provided in generate_text data."})
        return

    max_new_tokens = data.get('MaxNewTokens', 50)
    temperature = data.get('Temperature', 0.7)
    top_k = data.get('TopK', 40)
    top_p = data.get('TopP', 0.9)

    # --- Lógica de Geração (Inferência) e Streaming via WebSocket ---
    model.eval() # Colocar o modelo em modo de avaliação

    try:
        input_tokens = tokenizer.Encode(prompt_text, allow_special_tokens_in_text=False)
        generated_tokens = list(input_tokens)

        eos_token_id = tokenizer.GetEosTokenId()
        model_max_seq_len = model.MaxSeqLen

        print(f"Prompt para geração WS: '{prompt_text}'")

        with torch.no_grad(): # Desativar cálculo de gradientes para inferência
            for i in range(max_new_tokens):
                current_sequence_for_model = generated_tokens[-model_max_seq_len:]
                current_sequence_tensor = torch.tensor([current_sequence_for_model], dtype=torch.long, device=model.device)

                output = model(current_sequence_tensor) # [1, T_atual_real, vocab_size]
                logits = output[0, -1, :] # [vocab_size]

                next_token_id = SamplingUtils.sample_next_token( # SamplingUtils usado aqui
                    logits,
                    temperature,
                    top_k,
                    top_p
                )

                next_token_text = tokenizer.Decode([next_token_id])

                emit('generated_token', {'token': next_token_text}) # Emitir para o cliente

                if next_token_id == eos_token_id:  # <--- A condição de parada
                    print("Generated EOS token, stopping generation.")  # Log (no servidor)
                    break

                generated_tokens.append(next_token_id)

        print("Generation complete.")
        emit('generation_complete', {'final_seq_len': len(generated_tokens)}) # Emitir conclusão

    except Exception as e:
        print(f"Erro durante a geração WebSocket: {e}")
        # import traceback; traceback.print_exc() # Descomente para log detalhado
        emit('error', {'message': f"Error during generation: {e}"}) # Envia erro para o cliente


# --- Função auxiliar de Inferência (reutilizada pelo endpoint HTTP) ---
# Executa a geração de texto e retorna a string completa (não faz streaming)
def generate_text_inference(prompt_text: str, max_new_tokens: int, temperature: float, top_k: int, top_p: float,
                          model: TransformerModel, tokenizer: Tokenizer) -> str:
    """
    Executa a geração de texto (inferência) com o modelo e retorna a string completa gerada.
    Esta função não faz streaming.
    """
    # SamplingUtils é acessível (importado no topo)
    model.eval()

    input_tokens = tokenizer.Encode(prompt_text, allow_special_tokens_in_text=False)
    generated_tokens = list(input_tokens)

    eos_token_id = tokenizer.GetEosTokenId()
    model_max_seq_len = model.MaxSeqLen

    with torch.no_grad():
        for _ in range(max_new_tokens):
            current_sequence_for_model = generated_tokens[-model_max_seq_len:]
            current_sequence_tensor = torch.tensor([current_sequence_for_model], dtype=torch.long, device=model.device)

            output = model(current_sequence_tensor)
            logits = output[0, -1, :]

            next_token_id = SamplingUtils.sample_next_token( # SamplingUtils usado aqui
                logits,
                temperature,
                top_k,
                top_p
            )

            if next_token_id == eos_token_id:
                break

            generated_tokens.append(next_token_id)

    decoded_text = tokenizer.Decode(generated_tokens)

    # Opcional: retornar apenas a parte gerada
    # prompt_end_index = len(input_tokens)
    # decoded_text = tokenizer.Decode(generated_tokens[prompt_end_index:])

    return decoded_text


# --- PONTO DE EXECUÇÃO ---
# Este bloco é o que roda quando você executa `python app.py` diretamente
setup_application()
if __name__ == '__main__':
    print("\n--- Rodando Servidor de Desenvolvimento Flask com SocketIO ---")
    # Chame a função de setup para configurar o modelo/tokenizer


    # Agora rode o servidor SocketIO com as instâncias globais `app` e `socketio`
    # Use debug=True para ver logs detalhados
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)

