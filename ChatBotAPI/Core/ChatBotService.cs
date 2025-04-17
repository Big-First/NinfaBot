// ChatBotService.cs - AJUSTADO COM TEMPERATURE SAMPLING

using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch; // static using para torch

namespace ChatBotAPI.Core
{
    public class ChatBotService
    {
        private readonly TorchSharpModel model;
        private readonly Tokenizer tokenizer;
        private readonly Device device;
        private readonly int maxGeneratedTokens;
        private readonly int padTokenId;
        private readonly float samplingTemperature; // Novo: Temperatura para sampling

        public ChatBotService(TorchSharpModel model, Tokenizer tokenizer)
        {
            this.model = model ?? throw new ArgumentNullException(nameof(model));
            this.tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
            this.device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            this.model.to(this.device);
            this.padTokenId = this.tokenizer.PadTokenId;
            this.maxGeneratedTokens = 30; // Limite padrão
            this.samplingTemperature = 1.0f; // Temperatura padrão (1.0 = sem efeito, >1 aumenta aleatoriedade, <1 diminui)

            Console.WriteLine($"ChatBotService using device: {this.device.type}");
            Console.WriteLine($"ChatBotService configured with PadTokenId={this.padTokenId}, MaxGeneratedTokens={this.maxGeneratedTokens}, SamplingTemperature={this.samplingTemperature}");

            this.model.eval(); // Modo de avaliação
            Console.WriteLine("ChatBotService: Model set to eval() mode.");
        }

        public async Task ProcessMessage(WebSocket webSocket, string message)
        {
            if (string.IsNullOrWhiteSpace(message)) { /* ... */ return; }
            Console.WriteLine($"ChatBotService: === Processing message: '{message}' ===");

            Tensor? initialInputTensor = null;
            Tensor? currentInput = null;
            List<int> generatedTokenIds = new List<int>();
            string finalResponseMessage = "[Error: Generation failed]";

            try
            {
                // 1. Tokenizar Input Inicial e remover padding
                Console.WriteLine($"ChatBotService: Tokenizing initial input...");
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
                        Tensor? probabilities = null; // Para guardar as probabilidades
                        Tensor? predictedIndexTensor = null; // Para guardar o índice sorteado
                        long predictedTokenIdLong = -1;
                        int predictedTokenId = -1;

                        try
                        {
                            // 3a. Forward Pass
                            outputLogits = model.forward(currentInput);
                            if ((bool)(outputLogits == null)) { /* ... erro ... */ break; }

                            // ***** INÍCIO SAMPLING *****
                            // 3b. Aplicar Temperatura e Softmax
                             Console.WriteLine($"ChatBotService: Step {step+1}: Applying Temperature ({this.samplingTemperature}) and Softmax...");
                             // Divide logits pela temperatura (adiciona pequeno epsilon para evitar divisão por zero se T=0)
                             var scaledLogits = outputLogits / Math.Max(this.samplingTemperature, 1e-6f); // Garante T > 0
                             // Calcula as probabilidades
                             using (var tempProb = torch.softmax(scaledLogits, dim: 0)) // Softmax na dimensão do vocabulário
                             {
                                 probabilities = tempProb.clone(); // Clona para usar fora do using
                             }
                              Console.WriteLine($"ChatBotService: Step {step+1}: Probabilities calculated.");

                            // 3c. Sortear Próximo Token (Multinomial Sampling)
                            Console.WriteLine($"ChatBotService: Step {step+1}: Sampling next token using multinomial...");
                            // torch.multinomial espera probabilidades e retorna índices sorteados
                            predictedIndexTensor = torch.multinomial(probabilities, num_samples: 1);
                            predictedTokenIdLong = predictedIndexTensor.item<long>(); // Pega o ID sorteado
                            predictedTokenId = (int)predictedTokenIdLong;
                             Console.WriteLine($"ChatBotService: Step {step+1}: Sampled Token ID: {predictedTokenId}");
                            // ***** FIM SAMPLING *****


                            // 3d. Verificar Condição de Parada (PAD/UNK ID)
                            if (predictedTokenId == this.padTokenId)
                            {
                                Console.WriteLine($"ChatBotService: Step {step+1}: PAD/UNK token ({this.padTokenId}) sampled. Stopping generation.");
                                break; // Sai do loop for
                            }

                            // 3e. Adicionar token válido à resposta
                            generatedTokenIds.Add(predictedTokenId);

                            // 3f. Preparar Input para Próximo Passo (como antes)
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
                        finally
                        {
                            // Descarta tensores criados neste passo
                            outputLogits?.Dispose();
                            probabilities?.Dispose(); // Descarta tensor de probabilidades
                            predictedIndexTensor?.Dispose(); // Descarta tensor do índice
                        }
                    } // --- Fim do Loop FOR de Geração ---
                } // --- Fim do using no_grad ---

                // 4. Detokenizar a Sequência Gerada (como antes)
                Console.WriteLine($"ChatBotService: Generation finished. Detokenizing {generatedTokenIds.Count} tokens...");
                if (generatedTokenIds.Count > 0) {
                     finalResponseMessage = tokenizer.Detokenize(generatedTokenIds.ToArray());
                     Console.WriteLine($"ChatBotService: Final Detokenized Response: '{finalResponseMessage}'");
                } else { /* ... No response generated ... */ }
                if (string.IsNullOrWhiteSpace(finalResponseMessage)) { finalResponseMessage = "[Generated empty response]"; }

                // 5. Enviar Resposta Final (como antes)
                Console.WriteLine($"ChatBotService: Preparing to send final response: '{finalResponseMessage}'");
                await SendMessage(webSocket, finalResponseMessage);
                Console.WriteLine($"ChatBotService: SendMessage task awaited for final response.");

            } // Fim do Try Principal
            catch (Exception ex) { /* ... Log erro fatal externo ... */ }
            finally // Garante descarte dos tensores principais
            {
                 initialInputTensor?.Dispose();
                 currentInput?.Dispose();
                 Console.WriteLine($"ChatBotService: === Finished processing message: '{message}' ===");
            }
        }

        // Função auxiliar SendMessage (como antes)
        private async Task SendMessage(WebSocket webSocket, string message) { /* ... */ }

    } // Fim Classe
} // Fim Namespace