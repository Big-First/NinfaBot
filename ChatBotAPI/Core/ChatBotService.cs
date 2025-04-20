// ChatBotService.cs - Estrutura Corrigida, Sampling OK, Sem Refinamento

using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ChatBotAPI.Core
{
    public class ChatBotService
    {
        private readonly Module<Tensor, Tensor> model; // Aceita TransformerModel
        private readonly Tokenizer tokenizer;
        private readonly Device device;
        private readonly int maxGeneratedTokens;
        private readonly int padTokenId;
        private readonly float samplingTemperature;
        private readonly int topK;
        private readonly float topP;
        private readonly HashSet<int> eosTokenIds;

        public ChatBotService(
            Module<Tensor, Tensor> model,
            Tokenizer tokenizer,
            int maxGeneratedTokens,
            float samplingTemperature,
            int topK = 0,
            float topP = 0.0f)
        {
            this.model = model ?? throw new ArgumentNullException(nameof(model));
            this.tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));

            // *** CORREÇÃO: Obtém o device do primeiro parâmetro do modelo ***
            var firstParam = this.model.parameters().FirstOrDefault();
            if ((bool)(firstParam == null))
            {
                Console.WriteLine("Warning: Model has no parameters. Assuming CPU device for ChatBotService.");
                this.device = torch.CPU;
            }
            else
            {
                this.device = firstParam.device; // Pega o device do parâmetro
            }

            this.model.to(this.device);

            this.padTokenId = this.tokenizer.PadTokenId;
            this.eosTokenIds = new HashSet<int> { this.tokenizer.EosTokenId };

            this.maxGeneratedTokens = maxGeneratedTokens;
            this.samplingTemperature = Math.Max(samplingTemperature, 1e-6f);
            this.topK = topK;
            this.topP = topP;

            if (this.topK > 0 && this.topP > 0.0f && this.topP < 1.0f)
            {
                Console.WriteLine("Warning: Both Top-K and Top-P sampling are configured.");
            }

            Console.WriteLine($"ChatBotService using device: {this.device.type}");
            Console.WriteLine($"ChatBotService configured with:");
            Console.WriteLine($"  PadTokenId         = {this.padTokenId}");
            Console.WriteLine($"  EosTokenIds        = [{string.Join(", ", this.eosTokenIds)}]");
            Console.WriteLine($"  MaxGeneratedTokens = {this.maxGeneratedTokens}");
            Console.WriteLine($"  SamplingTemperature= {this.samplingTemperature}");
            Console.WriteLine($"  TopK               = {this.topK} {(this.topK <= 0 ? "(Disabled)" : "")}");
            Console.WriteLine(
                $"  TopP (Nucleus)     = {this.topP} {(this.topP <= 0.0f || this.topP >= 1.0f ? "(Disabled)" : "")}");

            this.model.eval();
            Console.WriteLine("ChatBotService: Model set to eval() mode.");
        }

        // --- MÉTODO ProcessMessage CORRIGIDO ---
        public async Task ProcessMessage(WebSocket webSocket, string message)
        {
            if (string.IsNullOrEmpty(message)) return;
            Console.WriteLine($"ChatBotService: === Processing message: '{message}' ===");

            Tensor? currentInput = null;
            List<int> generatedTokenIds = new List<int>();
            string finalResponseMessage = "[Error: Generation failed]";
            int lastPredictedTokenId = -1;

            try
            {
                // 1. Tokenizar e Preparar Tensor Inicial [1, SeqLen]
                int[] inputTokens = tokenizer.Tokenize(message);
                int[] initialSequence = inputTokens.Where(t => t != this.padTokenId).ToArray();
                if (initialSequence.Length == 0)
                {
                    await SendMessage(webSocket, "[Error: Invalid input]");
                    return;
                }

                long[] initialLongs = initialSequence.Select(id => (long)id).ToArray();
                using var tempInitialTensor = tensor(initialLongs, dtype: ScalarType.Int64).unsqueeze(0).to(device);
                currentInput = tempInitialTensor.clone().to(device);
                Console.WriteLine($"ChatBotService: Initial Input Tensor Shape: {currentInput.shape}");

                // 2. Loop de Geração
                model.eval();
                using (var noGrad = torch.no_grad())
                {
                    for (int step = 0; step < this.maxGeneratedTokens; step++)
                    {
                        Console.WriteLine($"ChatBotService: --> Generation Step {step + 1}/{this.maxGeneratedTokens}");

                        Tensor? outputLogitsFullSeq = null; // Logits da sequência completa
                        Tensor? lastTokenLogits = null; // Logits do último token [1, VocabSize]
                        Tensor? logits_batch0 = null; // Logits 1D [VocabSize]
                        Tensor? scaledLogits = null;
                        Tensor? finalLogitsForSampling = null;
                        Tensor? probabilities = null;
                        Tensor? predictedIndexTensor = null;
                        int predictedTokenId = -1;

                        try
                        {
                            // 2a. Forward Pass - Obtém logits para TODA a sequência
                            // Input: [1, CurrentSeqLen] -> Output: [CurrentSeqLen, 1, VocabSize] (assumindo batch_first=False no modelo)
                            outputLogitsFullSeq = model.forward(currentInput);

                            if ((bool)(outputLogitsFullSeq == null) || outputLogitsFullSeq.numel() == 0)
                            {
                                Console.Error.WriteLine("Error: Model forward returned null or empty logits sequence.");
                                break;
                            }

                            // Verifica se tem pelo menos 3 dimensões
                            if (outputLogitsFullSeq.shape.Length < 3)
                            {
                                Console.Error.WriteLine(
                                    $"Error: Model returned unexpected logits shape: {outputLogitsFullSeq.shape}. Expected [SeqLen, Batch, VocabSize].");
                                break;
                            }

                            // *** AJUSTE: Pega os logits APENAS DO ÚLTIMO TOKEN da sequência de saída ***
                            // Pega o último slice da dimensão 0 (SeqLen) -> Shape [1, VocabSize] (mantendo Batch dim)
                            lastTokenLogits = outputLogitsFullSeq[-1, .., ..];

                            // Pega os logits 1D do único item no batch para sampling
                            logits_batch0 = lastTokenLogits.squeeze(0); // Shape: [VocabSize]

                            // 2b. Penalidade de Repetição (opera em logits_batch0)
                            if (lastPredictedTokenId != -1 && lastPredictedTokenId >= 0 &&
                                lastPredictedTokenId < logits_batch0.shape[0])
                            {
                                logits_batch0[lastPredictedTokenId] = -float.MaxValue;
                            }

                            // 2c. Aplicar Temperatura (opera em logits_batch0)
                            scaledLogits = logits_batch0 / this.samplingTemperature;
                            finalLogitsForSampling = scaledLogits.clone();

                            // 2d. Aplicar Top-K (opera em finalLogitsForSampling 1D)
                            bool useTopK = this.topK > 0;
                            if (useTopK)
                            {
                                int k_actual = Math.Min(this.topK, (int)finalLogitsForSampling.shape[0]);
                                var topk_tuple = torch.topk(finalLogitsForSampling, k_actual, dim: 0);
                                using var filteredLogits = torch.full_like(finalLogitsForSampling, -float.MaxValue);
                                filteredLogits.scatter_(0, topk_tuple.indices, topk_tuple.values); // Usa scatter_
                                finalLogitsForSampling.Dispose();
                                finalLogitsForSampling = filteredLogits.clone();
                                filteredLogits.Dispose();
                            }

                            // 2e. Aplicar Top-P (opera em finalLogitsForSampling 1D)
                            bool useTopP = this.topP > 0.0f && this.topP < 1.0f;
                            if (useTopP)
                            {
                                Tensor? indices_to_remove = null;
                                try
                                {
                                    var sorted_logits_tuple =
                                        torch.sort(finalLogitsForSampling, dim: 0, descending: true);
                                    using var sorted_probs_p = torch.softmax(sorted_logits_tuple.values, dim: 0);
                                    using var cumulative_probs_p = torch.cumsum(sorted_probs_p, dim: 0);
                                    using var remove_mask = cumulative_probs_p > this.topP;
                                    if (remove_mask.shape[0] > 1) remove_mask[1..] = remove_mask[..^1].clone();
                                    remove_mask[0] = false;
                                    indices_to_remove = sorted_logits_tuple.indices.masked_select(remove_mask);
                                    if (indices_to_remove.numel() > 0)
                                        finalLogitsForSampling.index_fill_(0, indices_to_remove, -float.MaxValue);
                                }
                                finally
                                {
                                    indices_to_remove?.Dispose();
                                }
                            }

                            // 2f. Softmax e Sorteio (Multinomial)
                            probabilities = torch.softmax(finalLogitsForSampling, dim: 0);
                            // *** INÍCIO DEBUG EOS/TOP TOKENS ***
                            if (step > 3) // Verifica após alguns passos
                            {
                                try
                                {
                                    int eosId = this.tokenizer.EosTokenId;
                                    float eosProbability = probabilities[eosId].item<float>();
                                    int checkTopN = 5;
                                    var topTuple = torch.topk(probabilities, checkTopN, dim: 0);
                                    float[] topProbs = topTuple.values.cpu().data<float>().ToArray();
                                    int[] topIndices = topTuple.indices.cpu().data<long>().Select(i=>(int)i).ToArray();

                                    Console.WriteLine($"--- Step {step+1} Debug ---");
                                    Console.WriteLine($"  EOS Token ({eosId}) Probability: {eosProbability:P6}");
                                    Console.WriteLine($"  Top {checkTopN} Tokens:");
                                    for(int i = 0; i < Math.Min(checkTopN, topIndices.Length); i++) // Garante que não exceda os limites
                                    {
                                        string tokenStr = $"ID:{topIndices[i]}";
                                        try { tokenStr = tokenizer.Detokenize(new int[] { topIndices[i] }); if (string.IsNullOrWhiteSpace(tokenStr)) tokenStr = $"ID:{topIndices[i]} (Empty)"; } catch {}
                                        Console.WriteLine($"    - '{tokenStr}' (ID: {topIndices[i]}): {topProbs[i]:P6}");
                                    }
                                    if (useTopK || useTopP) {
                                        float eosLogitValue = finalLogitsForSampling[eosId].item<float>();
                                        bool eosWasConsidered = eosLogitValue > float.NegativeInfinity;
                                        // Ou: bool eosWasConsidered = eosLogitValue != -float.MaxValue; (se usou MaxValue no filtro)
                                        Console.WriteLine($"  EOS Token Considered (after K/P filter)? {eosWasConsidered} (Logit: {eosLogitValue})");
                                    }
                                    Console.WriteLine($"--------------------");
                                }
                                catch(Exception dbgEx) { Console.Error.WriteLine($"Error during EOS/Top token debug: {dbgEx.Message}"); }
                            }
                            // *** FIM DEBUG EOS/TOP TOKENS ***
                            if (probabilities.isnan().any().item<bool>())
                            {
                                /*...*/
                                break;
                            }
                            

                            predictedIndexTensor = torch.multinomial(probabilities, num_samples: 1);
                            predictedTokenId = (int)predictedIndexTensor.item<long>();
                            Console.WriteLine($"ChatBotService: Step {step + 1}: Sampled Token ID: {predictedTokenId}");

                            // 2g. Verificar EOS
                            if (this.eosTokenIds.Contains(predictedTokenId))
                            {
                                /*...*/
                                break;
                            }

                            // 2h. Adicionar Token e Preparar Próximo Input
                            generatedTokenIds.Add(predictedTokenId);
                            lastPredictedTokenId = predictedTokenId;
                            // (Lógica para preparar 'currentInput' permanece a mesma)
                            long[] previousInputLongs;
                            using (var currentInputCpu = currentInput.cpu())
                            using (var currentInputLong = currentInputCpu.to(ScalarType.Int64))
                            {
                                previousInputLongs = currentInputLong.squeeze(0).data<long>().ToArray();
                            }

                            var nextSequenceLongs = previousInputLongs.Concat(new long[] { (long)predictedTokenId })
                                .ToArray();
                            currentInput.Dispose();
                            if (nextSequenceLongs.Length > tokenizer.GetMaxSequenceLength())
                            {
                                /* Truncate */
                            }

                            currentInput = tensor(nextSequenceLongs, dtype: ScalarType.Int64).unsqueeze(0).to(device);
                        } // Fim Try interno do passo
                        catch (Exception stepEx)
                        {
                            Console.Error.WriteLine($"Error in generation step {step + 1}: {stepEx}");
                            break;
                        }
                        finally
                        {
                            // Dispose dos tensores DO PASSO
                            outputLogitsFullSeq?.Dispose(); // Descarta logits completos
                            lastTokenLogits?.Dispose(); // Descarta fatia do último token
                            logits_batch0?.Dispose(); // Descarta versão 1D
                            scaledLogits?.Dispose();
                            finalLogitsForSampling?.Dispose();
                            probabilities?.Dispose();
                            predictedIndexTensor?.Dispose();
                        }
                    } // --- Fim Loop FOR ---
                } // --- Fim using no_grad ---
                
                // ... (Detokenização DIRETA e envio como antes) ...
                PrintGeneratedTextDebug(generatedTokenIds, "Before Final Response Assignment");
            } // Fim Try Principal
            catch (Exception ex)
            {
                /*...*/
            }
            finally
            {
                /*...*/
            }
        } // --- Fim do ProcessMessage ---
        
        private string PrintGeneratedTextDebug(List<int> generatedTokenIds, string contextMessage = "Detokenized Debug Output")
        {
            Console.WriteLine($"--- DEBUG ({contextMessage}) ---");
            if (generatedTokenIds == null || !generatedTokenIds.Any())
            {
                Console.WriteLine("DEBUG: Token list empty/null.");
                Console.WriteLine("--- END DEBUG ---");
                return ""; // Retorna vazio se não há tokens
            }

            Console.WriteLine($"DEBUG: Trying to detokenize {generatedTokenIds.Count} tokens: [{string.Join(", ", generatedTokenIds)}]");
            string detokenizedText = "[Error in Debug Detokenization]"; // Default
            try
            {
                // Chama o Detokenize do nosso Tokenizer (que usa SharpToken)
                // Passa os IDs como array int[], pois Detokenize espera isso
                detokenizedText = tokenizer.Detokenize(generatedTokenIds.ToArray());

                if (string.IsNullOrWhiteSpace(detokenizedText))
                {
                    Console.WriteLine("DEBUG: Detokenized text is NULL or Whitespace.");
                    detokenizedText = "[Debug Detokenized Empty]"; // Indica que ficou vazio
                }
                else
                {
                    Console.WriteLine($"DEBUG: Detokenized Text: >>>{detokenizedText}<<<");
                }
            }
            catch (Exception ex)
            {
                 Console.Error.WriteLine($"DEBUG: ERROR during detokenization: {ex.ToString()}");
                 detokenizedText = "[Debug Detokenization Error]"; // Indica erro
            }
            finally
            {
                Console.WriteLine("--- END DEBUG ---");
            }
             // Retorna o texto detokenizado (ou mensagem de erro/vazio)
             // para que possa ser usado se necessário (embora não seja o caso agora)
            return detokenizedText;
        }
        // --- Fim da Função de Debug ---

        // --- Função auxiliar SendMessage ---
        private async Task SendMessage(WebSocket webSocket, string message)
        {
            /* ... */
        }
    } // --- FIM ChatBotService ---
} // --- FIM Namespace ---