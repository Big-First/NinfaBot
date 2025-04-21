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
                if (initialSequence.Length == 0) { await SendMessage(webSocket, "[Error: Invalid input]"); return; } // Simplificado
                long[] initialLongs = initialSequence.Select(id => (long)id).ToArray();
                initialInputTensor = tensor(initialLongs, dtype: ScalarType.Int64).to(device);
                currentInput = initialInputTensor.clone().to(device);

                // 3. Loop de Geração
                model.eval();
                using (var noGrad = torch.no_grad())
                {
                    // Use um limite maior para o teste de probabilidade EOS
                    int generationLimit = 100; // Aumente se necessário
                    Console.WriteLine($"ChatBotService: Starting generation loop (limit: {generationLimit} steps).");

                    for (int step = 0; step < generationLimit; step++) // Use o limite maior aqui
                    {
                        Tensor? outputLogits = null, scaledLogits = null, finalLogitsForSampling = null;
                        Tensor? probabilities = null, predictedIndexTensor = null;
                        int predictedTokenId = -1;

                        try
                        {
                            outputLogits = model.forward(currentInput);
                            if ((bool)(outputLogits == null) || outputLogits.numel() == 0)
                            {
                                Console.Error.WriteLine("Error: Model returned null or empty logits. Stopping generation.");
                                break;
                            }

                            // Penalidade Repetição (simples: último token)
                            if (lastPredictedTokenId != -1 && lastPredictedTokenId >= 0 && lastPredictedTokenId < outputLogits.shape[0])
                            {
                                outputLogits[lastPredictedTokenId] = -float.MaxValue;
                            }

                            // Aplicar Temperatura
                            scaledLogits = outputLogits / Math.Max(this.samplingTemperature, 1e-6f);

                            // Clonar para aplicar Top-K/Top-P sem modificar os logits escalados originais
                            // (Embora neste código simplificado, não estamos usando Top-K/P, manter o clone é boa prática)
                            finalLogitsForSampling = scaledLogits.clone();

                            // *** (Opcional: Reintroduza a lógica Top-K / Top-P aqui se desejar usá-la) ***
                            // Exemplo:
                            // // Aplicar Top-K
                            // bool useTopK = this.topK > 0;
                            // if (useTopK) { finalLogitsForSampling = ApplyTopK(finalLogitsForSampling, this.topK); }
                            // // Aplicar Top-P
                            // bool useTopP = this.topP > 0.0f && this.topP < 1.0f;
                            // if (useTopP) { finalLogitsForSampling = ApplyTopP(finalLogitsForSampling, this.topP); }

                            // Calcular Probabilidades FINAIS (após filtros, se houver)
                            probabilities = torch.softmax(finalLogitsForSampling, dim: 0);

                            // ----- INÍCIO: LOG DA PROBABILIDADE DO EOS -----
                            // *** CORREÇÃO: Usa cast explícito (bool) ***
                            if (this.eosTokenIds.Any() && (bool)probabilities)
                            {
                                // Assumindo que eosTokenIds contém o ID correto (50256)
                                int eosId = this.eosTokenIds.First();
                                if (eosId >= 0 && eosId < probabilities.shape[0])
                                {
                                    try
                                    {
                                        // .item<float>() funciona direto em tensores de 1 elemento, independente do device
                                        float eosProbability = probabilities[eosId].item<float>();
                                        // Log com alta precisão (8 casas decimais)
                                        Console.WriteLine($"      Step {step + 1}: Pr(EOS={eosId}) = {eosProbability:F8}");
                                    }
                                    catch (Exception probEx)
                                    {
                                        Console.Error.WriteLine($"      Step {step + 1}: Error getting probability for EOS={eosId}: {probEx.Message}");
                                    }
                                }
                                // else { // Log se ID EOS estiver fora dos limites - improvável }
                            }
                            // ----- FIM: LOG DA PROBABILIDADE DO EOS -----

                            // Amostragem Multinomial
                            predictedIndexTensor = torch.multinomial(probabilities, num_samples: 1);
                            predictedTokenId = (int)predictedIndexTensor.item<long>();
                            Console.WriteLine($"ChatBotService: Step {step+1}: Sampled Token ID: {predictedTokenId}"); // Log do token escolhido

                            // Verificar EOS para parar
                            if (this.eosTokenIds.Contains(predictedTokenId))
                            {
                                Console.WriteLine($"ChatBotService: Step {step+1}: EOS token ({predictedTokenId}) sampled. Stopping generation.");
                                break; // Sai do loop FOR
                            }

                            // Adicionar token gerado e preparar próximo input (como antes)
                            generatedTokenIds.Add(predictedTokenId);
                            lastPredictedTokenId = predictedTokenId; // Atualiza para a próxima penalidade

                            var nextInputTokenTensor = tensor(new long[] { (long)predictedTokenId }, dtype: ScalarType.Int64).to(device);
                            long[] previousInputData; using (var cpuTensor = currentInput.cpu()) { previousInputData = cpuTensor.data<long>().ToArray(); }
                            var nextSequenceLongs = previousInputData.Concat(new long[] { (long)predictedTokenId }).ToArray();
                            currentInput.Dispose(); // Dispose tensor antigo ANTES de verificar tamanho

                            // Truncar se necessário
                            if (nextSequenceLongs.Length > tokenizer.GetMaxSequenceLength())
                            {
                                int startIndex = nextSequenceLongs.Length - tokenizer.GetMaxSequenceLength();
                                nextSequenceLongs = nextSequenceLongs.Skip(startIndex).ToArray();
                            }

                            // Criar novo tensor de input
                            currentInput = tensor(nextSequenceLongs, dtype: ScalarType.Int64).to(device);
                            nextInputTokenTensor.Dispose(); // Dispose do tensor temporário

                        }
                        catch (Exception stepEx)
                        {
                            Console.Error.WriteLine($"Error in generation step {step + 1}: {stepEx}");
                            break; // Sai do loop em caso de erro no passo
                        }
                        finally
                        {
                            // Dispose seguro dos tensores criados dentro do try do passo
                            outputLogits?.Dispose();
                            scaledLogits?.Dispose();
                            finalLogitsForSampling?.Dispose(); // Dispose o clone também
                            probabilities?.Dispose();
                            predictedIndexTensor?.Dispose();
                        }
                    } // --- Fim Loop FOR ---
                } // --- Fim using no_grad ---

                Console.WriteLine($"ChatBotService: Generation loop finished. Generated {generatedTokenIds.Count} tokens.");

                // --- Detokenização DIRETA (Sem Refinamento) --- (Como na sua versão)
                if (generatedTokenIds.Any())
                {
                    try
                    {
                        finalResponseMessage = tokenizer.Detokenize(generatedTokenIds.ToArray());
                        Console.WriteLine($"DEBUG: Raw (Final) detokenized response: >>>{finalResponseMessage}<<<");
                        if (string.IsNullOrWhiteSpace(finalResponseMessage))
                        {
                            finalResponseMessage = "[No meaningful response generated]";
                        }
                    }
                    catch (Exception dtEx) { /*...*/ finalResponseMessage = "[Error processing response]"; }
                }
                else { /*...*/ finalResponseMessage = "[No response generated]"; }
                // --- Fim Detokenização ---

                // 4. Envia a resposta final
                await SendMessage(webSocket, finalResponseMessage);
                Console.WriteLine($"ChatBotService: SendMessage task awaited for final response: '{finalResponseMessage}'");

            } // Fim Try Principal
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Error in ProcessMessage: {ex.ToString()}");
                await SendMessage(webSocket, "[An internal error occurred]"); // Tenta enviar erro genérico
            }
            finally
            {
                initialInputTensor?.Dispose();
                currentInput?.Dispose(); // Garante dispose do último currentInput
            }
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