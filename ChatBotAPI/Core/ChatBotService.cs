// ChatBotService.cs - AJUSTADO COM DETECÇÃO DE EOS E PENALIDADE DE REPETIÇÃO E CORREÇÃO DE DETOKENIZAÇÃO

using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace ChatBotAPI.Core
{
    public class ChatBotService
    {
        private readonly TorchSharpModel model;
        private readonly Tokenizer tokenizer;
        private readonly Device device;
        private readonly int maxGeneratedTokens;
        private readonly int padTokenId;
        private readonly float samplingTemperature;
        private readonly HashSet<int> eosTokenIds; // IDs que indicam fim de sentença

        public ChatBotService(TorchSharpModel model, Tokenizer tokenizer)
        {
            this.model = model ?? throw new ArgumentNullException(nameof(model));
            this.tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
            // Ajuste aqui se quiser usar GPU se disponível, caso contrário, force CPU
            this.device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            // this.device = torch.CPU; // Linha para forçar CPU
            this.model.to(this.device);
            this.padTokenId = this.tokenizer.PadTokenId; // Geralmente ID 0
            this.maxGeneratedTokens = 30;
            this.samplingTemperature = 0.7f; // Mantendo moderado (ajuste se necessário)

            // --- Definir Tokens de Fim de Sentença ---
            // Inclui PAD/UNK (ID 0) e IDs para '.', '?', '!' (se existirem no tokenizer)
            this.eosTokenIds = new HashSet<int> { this.padTokenId };
            string[] commonEosStrings = { ".", "?", "!" };
            foreach (string eosStr in commonEosStrings)
            {
                // Usa o novo método TryGetTokenId
                if (tokenizer.TryGetTokenId(eosStr, out int id))
                {
                    this.eosTokenIds.Add(id);
                } else {
                    Console.WriteLine($"Warning: EOS token '{eosStr}' not found in tokenizer vocabulary.");
                }
            }
            // --- Fim Definição EOS ---

            Console.WriteLine($"ChatBotService using device: {this.device.type}");
            Console.WriteLine($"ChatBotService configured with PadTokenId={this.padTokenId}, MaxGeneratedTokens={this.maxGeneratedTokens}, SamplingTemperature={this.samplingTemperature}");
            Console.WriteLine($"ChatBotService EOS Token IDs: [{string.Join(", ", this.eosTokenIds)}]");

            this.model.eval();
            Console.WriteLine("ChatBotService: Model set to eval() mode.");
        }

        public async Task ProcessMessage(WebSocket webSocket, string message)
        {
            // Verifica se a mensagem é nula ou vazia
            if (string.IsNullOrWhiteSpace(message))
            {
                Console.WriteLine("ChatBotService: Received empty message.");
                await SendMessage(webSocket, "Please send a valid message.");
                return;
            }

            Console.WriteLine($"ChatBotService: === Processing message: '{message}' ===");

            Tensor? initialInputTensor = null;
            Tensor? currentInput = null;
            List<int> generatedTokenIds = new List<int>();
            string finalResponseMessage = "[Error: Generation failed]"; // Mantém default, será sobrescrito se houver tokens
            int lastPredictedTokenId = -1; // Para penalidade de repetição

            try
            {
                // 1. Tokenizar Input Inicial e remover padding
                int[] inputTokens = tokenizer.Tokenize(message);
                 if (inputTokens == null || !inputTokens.Any(t => t != this.padTokenId)) {
                     finalResponseMessage = "[Error: Input tokenization failed or resulted in only padding]";
                     Console.Error.WriteLine(finalResponseMessage);
                     await SendMessage(webSocket, finalResponseMessage);
                     return;
                 }
                 // Cria a sequência inicial removendo tokens de padding (ID 0) do final, se houver
                 // A tokenização já garante o padding no final, então podemos simplesmente considerar os tokens que não são PAD.
                 int actualInitialLength = inputTokens.TakeWhile(id => id != this.padTokenId).Count();
                 int[] initialSequence = inputTokens.Take(actualInitialLength).ToArray();
                 Console.WriteLine($"ChatBotService: Initial non-pad sequence length: {initialSequence.Length}");

                if (initialSequence.Length == 0)
                 {
                     finalResponseMessage = "[Error: Input resulted in empty sequence after removing padding]";
                     Console.Error.WriteLine(finalResponseMessage);
                     await SendMessage(webSocket, finalResponseMessage);
                     return;
                 }

                // 2. Preparar para Geração
                long[] initialLongs = initialSequence.Select(id => (long)id).ToArray();
                initialInputTensor = tensor(initialLongs, dtype: ScalarType.Int64, device: device);
                currentInput = initialInputTensor.clone().to(device); // Garante que está no device correto
                 Console.WriteLine($"ChatBotService: Initial Input Tensor Shape: {currentInput.shape}");

                // 3. Loop de Geração de Sequência
                model.eval(); // Garante que o modelo está em modo de avaliação
                using (var noGrad = torch.no_grad()) // Desabilita o cálculo de gradientes durante a inferência
                {
                    // Copia a sequência de entrada para a sequência atual para continuar a geração
                    var currentSequenceIds = new List<long>(initialLongs);

                    for (int step = 0; step < this.maxGeneratedTokens; step++)
                    {
                        // Console.WriteLine($"ChatBotService: --> Generation Step {step + 1}/{this.maxGeneratedTokens}"); // Log verbose

                        Tensor? outputLogits = null;
                        Tensor? probabilities = null;
                        Tensor? predictedIndexTensor = null;
                        long predictedTokenIdLong = -1;
                        int predictedTokenId = -1;

                        try
                        {
                            // 3a. Forward Pass
                            // O modelo espera um tensor com shape (SeqLen, BatchSize) ou (BatchSize, SeqLen) se batchFirst=true.
                            // Nosso modelo TorchSharpModel assume batchFirst=false e espera (SeqLen, BatchSize=1).
                            // Precisamos ajustar o shape do currentInput.
                             // currentInput já está no device e tem shape (SeqLen) no primeiro passo.
                             // Nos passos subsequentes, ele terá (SeqLen).
                             // Precisamos adicionar a dimensão BatchSize=1.
                             Tensor inputForModel = currentInput.unsqueeze(1); // Shape agora é (SeqLen, 1)

                            outputLogits = model.forward(inputForModel); // outputLogits shape deve ser (VocabSize)
                            inputForModel.Dispose(); // Limpa o tensor intermediário

                            if ((bool)(outputLogits == null)) {
                                Console.Error.WriteLine("ChatBotService: Model forward returned null logits. Stopping generation.");
                                break;
                            }

                            // ***** INÍCIO PENALIDADE DE REPETIÇÃO SIMPLES *****
                            // Aplica penalidade ao tensor de logits antes do softmax
                            if (lastPredictedTokenId != -1 && lastPredictedTokenId != this.padTokenId) // Não penaliza PAD/UNK se for o último gerado
                            {
                                // Reduz drasticamente a probabilidade de gerar o mesmo token novamente
                                // Atribuindo um valor muito baixo ao logit correspondente
                                if (lastPredictedTokenId < outputLogits.shape[0]) // Garante que o ID está dentro dos limites do vocabulário
                                {
                                    outputLogits[lastPredictedTokenId] = -float.MaxValue; // Ou um negativo grande: -1e9f;
                                    // Console.WriteLine($"DEBUG: Penalized repetition of token ID {lastPredictedTokenId}");
                                } else {
                                     Console.WriteLine($"Warning: Cannot apply repetition penalty for ID {lastPredictedTokenId} as it's out of vocab bounds ({outputLogits.shape[0]}).");
                                }
                            }
                            // ***** FIM PENALIDADE DE REPETIÇÃO SIMPLES *****


                            // 3b. Aplicar Temperatura e Softmax
                            // Logits tem shape (VocabSize). Softmax deve ser aplicado sobre a dimensão dos logits.
                            var scaledLogits = outputLogits / Math.Max(this.samplingTemperature, 1e-6f); // Evita divisão por zero
                            using (var tempProb = torch.softmax(scaledLogits, dim: 0)) {
                                probabilities = tempProb.clone(); // probabilities shape é (VocabSize)
                            }
                            scaledLogits.Dispose();


                            // 3c. Sortear Próximo Token
                            predictedIndexTensor = torch.multinomial(probabilities, num_samples: 1); // Retorna um tensor com 1 elemento [ID]
                            predictedTokenIdLong = predictedIndexTensor.item<long>(); // Extrai o ID como long
                            predictedTokenId = (int)predictedTokenIdLong; // Converte para int
                            // Console.WriteLine($"ChatBotService: Step {step+1}: Sampled Token ID: {predictedTokenId}"); // Log verbose


                            // ***** VERIFICAÇÃO DE EOS TOKEN *****
                            if (this.eosTokenIds.Contains(predictedTokenId))
                            {
                                Console.WriteLine($"ChatBotService: Step {step+1}: EOS token ({predictedTokenId}) sampled. Stopping generation.");
                                break; // Sai do loop for de geração
                            }
                            // ***** FIM VERIFICAÇÃO EOS *****


                            // 3d. Adicionar token válido à lista de IDs gerados
                            // A lista `generatedTokenIds` guarda APENAS os tokens da resposta gerada.
                            generatedTokenIds.Add(predictedTokenId);
                            lastPredictedTokenId = predictedTokenId; // Atualiza o último token gerado

                            // 3e. Preparar Input para Próximo Passo
                            // A nova entrada para o modelo é a sequência anterior + o token recém-gerado.
                            // currentSequenceIds agora armazena a sequência completa (input original + gerados).
                            currentSequenceIds.Add(predictedTokenIdLong);

                            // Truncar a sequência se ela exceder o tamanho máximo esperado pelo modelo
                            int maxSeqLen = tokenizer.GetMaxSequenceLength();
                             if (currentSequenceIds.Count > maxSeqLen) {
                                 // Mantém apenas os últimos maxSeqLen tokens
                                currentSequenceIds = currentSequenceIds.Skip(currentSequenceIds.Count - maxSeqLen).ToList();
                             }

                            // Cria o novo tensor de entrada para o próximo passo
                             currentInput.Dispose(); // Descarta o tensor anterior
                            currentInput = tensor(currentSequenceIds.ToArray(), dtype: ScalarType.Int64, device: device);


                        } // Fim do Try interno do loop
                        catch (Exception stepEx) {
                            Console.Error.WriteLine($"ChatBotService: ERROR during generation step {step+1}: {stepEx.ToString()}");
                            // Lançar ou quebrar dependendo da severidade
                            break; // Quebra o loop de geração em caso de erro
                        }
                        finally {
                            // Dispor tensores criados neste passo, exceto currentInput que é usado na próxima iteração
                            // outputLogits e probabilities precisam ser dispostos
                            outputLogits?.Dispose();
                            probabilities?.Dispose();
                            predictedIndexTensor?.Dispose();
                        }

                    } // --- Fim do Loop FOR de Geração ---
                } // --- Fim do using no_grad ---

                // 4. Detokenizar IDs Gerados e Preparar Resposta Final
                 Console.WriteLine($"ChatBotService: Generation finished. Detokenizing {generatedTokenIds.Count} tokens...");
                 if (generatedTokenIds.Count > 0)
                 {
                     // *** CORREÇÃO: Chama o Detokenize e atribui o resultado ***
                     string detokenizedResponse = tokenizer.Detokenize(generatedTokenIds.ToArray());
                     finalResponseMessage = detokenizedResponse; // Atribui a resposta detokenizada
                     Console.WriteLine($"ChatBotService: Detokenized response: '{finalResponseMessage}'");
                 }
                 else
                 {
                     // Se nenhum token foi gerado (por exemplo, input vazio, ou loop quebrou cedo)
                     finalResponseMessage = "[Error: No response tokens generated]"; // Mensagem de erro mais específica
                     Console.Error.WriteLine(finalResponseMessage);
                 }


                // 5. Enviar Resposta via WebSocket
                 Console.WriteLine($"ChatBotService: Preparing to send final response: '{finalResponseMessage}'");
                 await SendMessage(webSocket, finalResponseMessage);
                 Console.WriteLine($"ChatBotService: SendMessage task awaited for final response.");

            } // Fim do Try Principal
            catch (Exception ex) {
                Console.Error.WriteLine($"ChatBotService: FATAL ERROR processing message '{message}': {ex.ToString()}");
                 // Em caso de erro fatal, garante que o cliente receba alguma notificação de erro
                 if (webSocket.State == WebSocketState.Open) {
                     await SendMessage(webSocket, $"[Fatal Error: {ex.Message}]");
                 }
            }
            finally {
                // Dispor tensores principais alocados no Try (initialInputTensor e currentInput)
                initialInputTensor?.Dispose();
                currentInput?.Dispose();
                 Console.WriteLine($"ChatBotService: Disposed principal tensors.");
            }
        }

        // Função auxiliar SendMessage (como antes)
        private async Task SendMessage(WebSocket webSocket, string message)
        {
            if (webSocket.State == WebSocketState.Open)
            {
                var bytes = Encoding.UTF8.GetBytes(message);
                await webSocket.SendAsync(new ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, CancellationToken.None);
            }
            else
            {
                Console.WriteLine($"Warning: Tried to send message on closed WebSocket (State: {webSocket.State})");
            }
        }

        // --- FIM ChatBotService ---
    }

    // --- ADICIONAR MÉTODO AO TOKENIZER.CS (Se ainda não adicionou) ---
    // Você precisará adicionar este método à sua classe Tokenizer se não estiver lá:
    /*
    public partial class Tokenizer // Use partial se Tokenizer estiver em outro arquivo
    {
        // Tenta obter o ID de um token específico
        public bool TryGetTokenId(string token, out int id)
        {
            return wordToIndex.TryGetValue(token, out id);
        }
    }
    */
    // --- FIM ADIÇÃO AO TOKENIZER.CS ---
}