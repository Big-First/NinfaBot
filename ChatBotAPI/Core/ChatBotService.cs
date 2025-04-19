// ChatBotService.cs - Removido RefineGeneratedResponse, usa detokenização bruta

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
        private readonly int topK;
        private readonly float topP;
        private readonly HashSet<int> eosTokenIds;

        // Construtor (aceita parâmetros de sampling)
        public ChatBotService(
            TorchSharpModel model,
            Tokenizer tokenizer,
            int maxGeneratedTokens,
            float samplingTemperature,
            int topK = 0,
            float topP = 0.0f)
        {
            this.model = model ?? throw new ArgumentNullException(nameof(model));
            this.tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
            this.device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            this.model.to(this.device);

            this.padTokenId = this.tokenizer.PadTokenId;
            this.eosTokenIds = new HashSet<int> { this.tokenizer.EosTokenId }; // Apenas EOS 50256

            this.maxGeneratedTokens = maxGeneratedTokens;
            this.samplingTemperature = Math.Max(samplingTemperature, 1e-6f);
            this.topK = topK;
            this.topP = topP;

             if (this.topK > 0 && this.topP > 0.0f && this.topP < 1.0f) { /* Log Warning */ } // Aviso opcional

            Console.WriteLine($"ChatBotService using device: {this.device.type}");
            Console.WriteLine($"ChatBotService configured with:");
            Console.WriteLine($"  PadTokenId         = {this.padTokenId}");
            Console.WriteLine($"  EosTokenIds        = [{string.Join(", ", this.eosTokenIds)}]");
            Console.WriteLine($"  MaxGeneratedTokens = {this.maxGeneratedTokens}");
            Console.WriteLine($"  SamplingTemperature= {this.samplingTemperature}");
            Console.WriteLine($"  TopK               = {this.topK} {(this.topK <= 0 ? "(Disabled)" : "")}");
            Console.WriteLine($"  TopP (Nucleus)     = {this.topP} {(this.topP <= 0.0f || this.topP >= 1.0f ? "(Disabled)" : "")}");

            this.model.eval();
            Console.WriteLine("ChatBotService: Model set to eval() mode.");
        }

        // --- Função de Debug (pode ser mantida ou removida se não for mais útil) ---
        private string PrintGeneratedTextDebug(List<int> generatedTokenIds, string contextMessage = "Detokenized Debug Output")
        {
            Console.WriteLine($"--- DEBUG ({contextMessage}) ---");
            if (generatedTokenIds == null || !generatedTokenIds.Any()) { Console.WriteLine("DEBUG: Token list empty/null."); Console.WriteLine("--- END DEBUG ---"); return ""; }
            Console.WriteLine($"DEBUG: Trying to detokenize {generatedTokenIds.Count} tokens: [{string.Join(", ", generatedTokenIds)}]");
            try
            {
                string detokenizedText = tokenizer.Detokenize(generatedTokenIds.ToArray());
                Console.WriteLine($"DEBUG: Detokenized Text: >>>{detokenizedText}<<<"); // Log sempre, mesmo se vazio
                return detokenizedText;
            }
            catch (Exception ex) { Console.Error.WriteLine($"DEBUG: Error during detokenization: {ex.Message}");}
            finally { Console.WriteLine("--- END DEBUG ---"); }
            return "";
        }

        // --- MÉTODO ProcessMessage SEM REFINAMENTO ---
        public async Task ProcessMessage(WebSocket webSocket, string message)
        {
            if (string.IsNullOrEmpty(message)) return;
            Console.WriteLine($"ChatBotService: === Processing message: '{message}' ===");

            Tensor? initialInputTensor = null;
            Tensor? currentInput = null;
            List<int> generatedTokenIds = new List<int>();
            string finalResponseMessage = "[Error: Generation failed]";
            int lastPredictedTokenId = -1;

            try
            {
                // 1. Tokenizar Input e 2. Preparar Tensor Inicial (como antes)
                int[] inputTokens = tokenizer.Tokenize(message);
                int[] initialSequence = inputTokens.Where(t => t != this.padTokenId).ToArray();
                if (initialSequence.Length == 0) { /*...*/ await SendMessage(webSocket, "[Error: Invalid input]"); return; }
                long[] initialLongs = initialSequence.Select(id => (long)id).ToArray();
                initialInputTensor = tensor(initialLongs, dtype: ScalarType.Int64).to(device);
                currentInput = initialInputTensor.clone().to(device);

                // 3. Loop de Geração (com sampling Temp/K/P)
                model.eval();
                using (var noGrad = torch.no_grad())
                {
                    for (int step = 0; step < this.maxGeneratedTokens; step++)
                    {
                        Tensor? outputLogits = null, scaledLogits = null, finalLogitsForSampling = null;
                        Tensor? probabilities = null, predictedIndexTensor = null;
                        int predictedTokenId = -1;

                        try
                        {
                            outputLogits = model.forward(currentInput);
                            if ((bool)(outputLogits == null) || outputLogits.numel() == 0) break;
                            if (lastPredictedTokenId != -1 && lastPredictedTokenId >= 0 && lastPredictedTokenId < outputLogits.shape[0]) { outputLogits[lastPredictedTokenId] = -float.MaxValue; } // Penalidade Repetição
                            scaledLogits = outputLogits / Math.Max(this.samplingTemperature, 1e-6f);
                            finalLogitsForSampling = scaledLogits.clone();

                            // Aplicar Top-K
                            bool useTopK = this.topK > 0;
                            if (useTopK) { /* ... lógica Top-K como antes ... */ }
                            // Aplicar Top-P
                            bool useTopP = this.topP > 0.0f && this.topP < 1.0f;
                            if (useTopP) { /* ... lógica Top-P como antes ... */ }

                            // Softmax e Multinomial
                            probabilities = torch.softmax(finalLogitsForSampling, dim: 0);
                            predictedIndexTensor = torch.multinomial(probabilities, num_samples: 1);
                            predictedTokenId = (int)predictedIndexTensor.item<long>();
                             Console.WriteLine($"ChatBotService: Step {step+1}: Sampled Token ID: {predictedTokenId}");


                            // Verificar APENAS EOS (50256)
                            if (this.eosTokenIds.Contains(predictedTokenId))
                            {
                                Console.WriteLine($"ChatBotService: Step {step+1}: EOS token ({predictedTokenId}) sampled. Stopping generation.");
                                break;
                            }

                            generatedTokenIds.Add(predictedTokenId);
                            lastPredictedTokenId = predictedTokenId;
                            // Preparar Próximo Input (lógica como antes)
                             var nextInputTokenTensor = tensor(new long[] { (long)predictedTokenId }, dtype: ScalarType.Int64).to(device);
                             long[] previousInputData; using (var cpuTensor = currentInput.cpu()) { previousInputData = cpuTensor.data<long>().ToArray(); }
                             var nextSequenceLongs = previousInputData.Concat(new long[] { (long)predictedTokenId }).ToArray(); currentInput.Dispose();
                             if (nextSequenceLongs.Length > tokenizer.GetMaxSequenceLength()) { int si = nextSequenceLongs.Length - tokenizer.GetMaxSequenceLength(); nextSequenceLongs = nextSequenceLongs.Skip(si).ToArray(); }
                             currentInput = tensor(nextSequenceLongs, dtype: ScalarType.Int64).to(device); nextInputTokenTensor.Dispose();
                        }
                        catch (Exception stepEx) { Console.Error.WriteLine($"Error in generation step {step + 1}: {stepEx}"); break; }
                        finally { /* ... Dispose tensores do passo ... */ }
                    } // --- Fim Loop FOR ---
                } // --- Fim using no_grad ---

                Console.WriteLine($"ChatBotService: Generation loop finished. Generated {generatedTokenIds.Count} tokens.");

                // --- Detokenização DIRETA (Sem Refinamento) ---
                if (generatedTokenIds.Any())
                {
                    try
                    {
                        // *** Usa diretamente o resultado do Detokenize ***
                        finalResponseMessage = tokenizer.Detokenize(generatedTokenIds.ToArray());
                        Console.WriteLine($"DEBUG: Raw (Final) detokenized response: >>>{finalResponseMessage}<<<");

                        // *** REMOVIDA a chamada para RefineGeneratedResponse ***

                        if (string.IsNullOrWhiteSpace(finalResponseMessage))
                        {
                            Console.WriteLine("ChatBotService: Final response message is empty/whitespace after detokenization.");
                            finalResponseMessage = "[No meaningful response generated]";
                        }
                    }
                    catch (Exception dtEx)
                    {
                        Console.Error.WriteLine($"Error during Detokenization: {dtEx.ToString()}");
                        finalResponseMessage = "[Error processing response]";
                    }
                }
                else
                {
                    Console.WriteLine("ChatBotService: No tokens were generated.");
                    finalResponseMessage = "[No response generated]";
                }
                // --- Fim Detokenização ---

                // 4. Envia a resposta final (bruta detokenizada)
                await SendMessage(webSocket, finalResponseMessage);
                Console.WriteLine($"ChatBotService: SendMessage task awaited for final response: '{finalResponseMessage}'");

            } // Fim Try Principal
            catch (Exception ex) { /* ... Log erro ... */ }
            finally { /* ... Dispose tensores principais ... */ }
        } // --- Fim do ProcessMessage ---


        // *** FUNÇÃO RefineGeneratedResponse REMOVIDA COMPLETAMENTE DA CLASSE ***


        // --- Função auxiliar SendMessage ---
        private async Task SendMessage(WebSocket webSocket, string message)
        {
            var messageBuffer = Encoding.UTF8.GetBytes(message);
            var segment = new ArraySegment<byte>(messageBuffer);
            await webSocket.SendAsync(segment, WebSocketMessageType.Text, true, CancellationToken.None);
        }

    } // --- FIM ChatBotService ---
} // --- FIM Namespace ---