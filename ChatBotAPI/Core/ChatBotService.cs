// ChatBotService.cs - AJUSTADO COM DETECÇÃO DE EOS E PENALIDADE DE REPETIÇÃO

using System.Net.WebSockets;
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
            this.device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            this.model.to(this.device);
            this.padTokenId = this.tokenizer.PadTokenId; // Geralmente ID 0
            this.maxGeneratedTokens = 30;
            this.samplingTemperature = 0.7f; // Mantendo moderado (ajuste se necessário)

            // --- Definir Tokens de Fim de Sentença ---
            // Inclui PAD/UNK (ID 0) e IDs para '.', '?', '!' (se existirem no tokenizer)
            this.eosTokenIds = new HashSet<int> { this.padTokenId };
            string[] commonEosStrings = { ".", "?", "!" };
            // --- Fim Definição EOS ---


            Console.WriteLine($"ChatBotService using device: {this.device.type}");
            Console.WriteLine($"ChatBotService configured with PadTokenId={this.padTokenId}, MaxGeneratedTokens={this.maxGeneratedTokens}, SamplingTemperature={this.samplingTemperature}");
            Console.WriteLine($"ChatBotService EOS Token IDs: [{string.Join(", ", this.eosTokenIds)}]");

            this.model.eval();
            Console.WriteLine("ChatBotService: Model set to eval() mode.");
        }
        
        private string PrintGeneratedTextDebug(List<int> generatedTokenIds, string contextMessage = "Detokenized Debug Output")
        {
            Console.WriteLine($"--- DEBUG ({contextMessage}) ---");
            if (generatedTokenIds == null || !generatedTokenIds.Any())
            {
                Console.WriteLine("DEBUG: Lista de tokens gerados está vazia ou nula.");
                Console.WriteLine("--- END DEBUG ---");
                return "";
            }

            Console.WriteLine($"DEBUG: Tentando detokenizar {generatedTokenIds.Count} tokens: [{string.Join(", ", generatedTokenIds)}]");

            try
            {
                // Usa o tokenizer injetado na ChatBotService (_tokenizer)
                // Certifique-se de que o nome da variável do tokenizer esteja correto (pode ser _tokenizer, tokenizer, etc.)
                string detokenizedText = tokenizer.Detokenize(generatedTokenIds.ToArray()); // Detokenize espera int[]

                if (string.IsNullOrWhiteSpace(detokenizedText))
                {
                    Console.WriteLine("DEBUG: Resultado da detokenização é VAZIO ou ESPAÇOS EM BRANCO.");
                }
                else
                {
                    Console.WriteLine($"DEBUG: Texto Detokenizado: >>>{detokenizedText}<<<");
                    return detokenizedText;
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"DEBUG: ERRO durante a detokenização para debug: {ex.Message}");
                Console.Error.WriteLine($"DEBUG: StackTrace: {ex.StackTrace}");
            }
            finally
            {
                Console.WriteLine("--- END DEBUG ---");
            }

            return "";
        }

        public async Task ProcessMessage(WebSocket webSocket, string message)
        {
            // ... (verificação de mensagem vazia) ...
            Console.WriteLine($"ChatBotService: === Processing message: '{message}' ===");

            Tensor? initialInputTensor = null;
            Tensor? currentInput = null;
            List<int> generatedTokenIds = new List<int>();
            string finalResponseMessage = "[Error: Generation failed]";
            int lastPredictedTokenId = -1; // Para penalidade de repetição

            try
            {
                // 1. Tokenizar Input Inicial e remover padding
                // ... (como antes) ...
                 int[] inputTokens = tokenizer.Tokenize(message);
                 if (inputTokens == null || !inputTokens.Any(t => t != this.padTokenId)) { /* ... erro ... */ return; }
                 int[] initialSequence = inputTokens.Where(t => t != this.padTokenId).ToArray();
                 Console.WriteLine($"ChatBotService: Initial non-pad sequence length: {initialSequence.Length}");

                // 2. Preparar para Geração
                long[] initialLongs = initialSequence.Select(id => (long)id).ToArray();
                initialInputTensor = tensor(initialLongs, dtype: ScalarType.Int64).to(device);
                currentInput = initialInputTensor.clone().to(device);
                 Console.WriteLine($"ChatBotService: Initial Input Tensor Shape: {currentInput.shape}");

                // 3. Loop de Geração de Sequência
                model.eval();
                using (var noGrad = torch.no_grad())
                {
                    for (int step = 0; step < this.maxGeneratedTokens; step++)
                    {
                        Console.WriteLine($"ChatBotService: --> Generation Step {step + 1}/{this.maxGeneratedTokens}");

                        Tensor? outputLogits = null;
                        Tensor? probabilities = null;
                        Tensor? predictedIndexTensor = null;
                        long predictedTokenIdLong = -1;
                        int predictedTokenId = -1;

                        try
                        {
                            // 3a. Forward Pass
                            outputLogits = model.forward(currentInput);
                            if ((bool)(outputLogits == null)) { break; }

                            // ***** INÍCIO PENALIDADE DE REPETIÇÃO SIMPLES *****
                            if (lastPredictedTokenId != -1) // Se não for o primeiro token gerado
                            {
                                // Reduz drasticamente a probabilidade de gerar o mesmo token novamente
                                // Atribuindo um valor muito baixo ao logit correspondente
                                outputLogits[lastPredictedTokenId] = -float.MaxValue; // Ou um negativo grande: -1e9f;
                                // Console.WriteLine($"DEBUG: Penalized repetition of token ID {lastPredictedTokenId}");
                            }
                            // ***** FIM PENALIDADE DE REPETIÇÃO SIMPLES *****


                            // 3b. Aplicar Temperatura e Softmax
                            var scaledLogits = outputLogits / Math.Max(this.samplingTemperature, 1e-6f);
                            using (var tempProb = torch.softmax(scaledLogits, dim: 0)) { probabilities = tempProb.clone(); }


                            // 3c. Sortear Próximo Token
                            predictedIndexTensor = torch.multinomial(probabilities, num_samples: 1);
                            predictedTokenIdLong = predictedIndexTensor.item<long>();
                            predictedTokenId = (int)predictedTokenIdLong;
                             Console.WriteLine($"ChatBotService: Step {step+1}: Sampled Token ID: {predictedTokenId}");


                            // ***** VERIFICAÇÃO DE EOS TOKEN *****
                            if (this.eosTokenIds.Contains(predictedTokenId))
                            {
                                Console.WriteLine($"ChatBotService: Step {step+1}: EOS token ({predictedTokenId}) sampled. Stopping generation.");
                                break; // Sai do loop for
                            }
                            // ***** FIM VERIFICAÇÃO EOS *****


                            // 3d. Adicionar token válido à resposta
                            generatedTokenIds.Add(predictedTokenId);
                            lastPredictedTokenId = predictedTokenId; // Atualiza o último token para a próxima iteração da penalidade

                            // 3e. Preparar Input para Próximo Passo (como antes)
                             // ... (concatenação, truncamento, dispose antigo, atribui novo currentInput) ...
                             var previousInputData = currentInput.to(CPU).data<long>().ToArray();
                             var nextSequence = previousInputData.Concat(new long[] { predictedTokenIdLong }).ToArray();
                             if (nextSequence.Length > tokenizer.GetMaxSequenceLength()) {
                                 nextSequence = nextSequence.Skip(nextSequence.Length - tokenizer.GetMaxSequenceLength()).ToArray();
                             }
                             var nextInputTensor = tensor(nextSequence, dtype: ScalarType.Int64).to(device);
                             currentInput.Dispose();
                             currentInput = nextInputTensor;

                        }
                        catch (Exception stepEx) { /* ... log erro fatal interno ... */ break; }
                        finally { /* ... dispose tensores do passo ... */ }

                    } // --- Fim do Loop FOR de Geração ---
                } // --- Fim do using no_grad ---
                Console.WriteLine($"ChatBotService: Generation finished. Detokenizing {generatedTokenIds.Count} tokens...");
                string response =  PrintGeneratedTextDebug(generatedTokenIds, "Após Loop de Geração");

               // ... (Detokenizar, tratar resposta vazia, enviar como antes) ...
                 Console.WriteLine($"ChatBotService: Generation finished. Detokenizing {generatedTokenIds.Count} tokens...");
                 if (generatedTokenIds.Count > 0) { /* ... Detokenize ... */ }
                 else { /* ... No response generated ... */ }
                 if (string.IsNullOrWhiteSpace(finalResponseMessage)) { /* ... Generated empty ... */ }
                 await SendMessage(webSocket, response);
                 Console.WriteLine($"ChatBotService: SendMessage task awaited for final response.");

            } // Fim do Try Principal
            catch (Exception ex) { /* ... Log erro fatal externo ... */ }
            finally { /* ... Dispose tensores principais ... */ }
        }

        // Função auxiliar SendMessage (como antes)
        private async Task SendMessage(WebSocket webSocket, string message) { /* ... */ }

        // --- FIM ChatBotService ---
    }

    // --- ADICIONAR MÉTODO AO TOKENIZER.CS ---
    // Você precisará adicionar este método à sua classe Tokenizer
    // para que o ChatBotService possa encontrar os IDs dos tokens EOS.
    // --- FIM ADIÇÃO AO TOKENIZER.CS ---
}