// Trainer.cs

using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using System.IO;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.nn;

namespace ChatBotAPI.Core
{
    public class Trainer
    {
        private readonly TorchSharpModel model;
        private readonly Tokenizer tokenizer;
        private readonly Optimizer optimizer;
        private readonly Module<Tensor, Tensor, Tensor> lossFunction;
        private readonly Device device;
        private readonly string modelSavePath;
        // Adiciona um helper para reportar progresso via IProgress<T> se necessário
        private Action<string> Report = Console.WriteLine; // Default para Console

        // Construtor MODIFICADO
        public Trainer(TorchSharpModel model, Tokenizer tokenizer, double learningRate, string modelSavePath)
        {
            this.model = model ?? throw new ArgumentNullException(nameof(model));
            this.tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
            this.device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            this.modelSavePath = Path.GetFullPath(modelSavePath);
            Console.WriteLine($"Trainer using device: {this.device.type}");
            Console.WriteLine($"Trainer configured to save model to: {this.modelSavePath}");
            this.model.to(this.device);
            // Adicionar weight decay se estiver usando para regularização
            this.optimizer = Adam(this.model.parameters(), lr: learningRate /*, weight_decay: 1e-5 */);

            // ***** MUDANÇA 1: Remover ignore_index da Loss Function *****
            this.lossFunction = CrossEntropyLoss().to(this.device);
            Console.WriteLine($"Trainer initialized loss function (CrossEntropyLoss). Device: {this.device.type}. NO ignore_index set.");
            // ***** FIM MUDANÇA 1 *****

            if (tokenizer.ActualVocabSize <= 2) { /* ... alerta ... */ }
        }

        // Método Train MODIFICADO para aceitar IProgress e usar Tokenize sem padding
        public void Train(List<string> trainingData, int epochs, IProgress<string>? progressReporter = null)
        {
             // Atualiza o reporter se um foi fornecido
            if (progressReporter != null) {
                Report = message => { Console.WriteLine(message); progressReporter.Report(message); };
            } else {
                 Report = Console.WriteLine; // Garante que Report não seja nulo
            }


            if (trainingData == null || !trainingData.Any()) { Report("Training data is empty. Skipping training."); return; }
            if (tokenizer.ActualVocabSize <= 0) { Report("ERROR: Cannot train: Tokenizer vocabulary size is invalid."); return; }

            // Não precisamos mais de padTokenId aqui para skip
            int vocabSize = tokenizer.ActualVocabSize;
            // maxSeqLen ainda pode ser útil para logs ou verificações, mas não para padding aqui
            int maxSeqLen = tokenizer.GetMaxSequenceLength();

            Report($"Starting TorchSharp training on {device.type}. Epochs: {epochs}. Sentences: {trainingData.Count}. VocabSize: {vocabSize}. MaxSeqLen: {maxSeqLen}. EOS Token WILL be trained.");
            Stopwatch epochStopwatch = new Stopwatch();
            Stopwatch totalStopwatch = Stopwatch.StartNew();

            // --- Loop Principal de Épocas ---
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                model.train();
                epochStopwatch.Restart();
                Report($"--- Epoch {epoch + 1}/{epochs} ---");

                float totalLoss = 0f;
                long totalStepsInEpoch = 0;
                long skippedSteps = 0; // Continuará contando skips de sequências curtas
                long sentenceCount = 0;
                long reportIntervalSteps = Math.Max(1, trainingData.Count / 5); // Log mais frequente com menos dados

                // Calcula o total estimado de passos UMA VEZ por época (mais eficiente)
                long estimatedTotalStepsInEpoch = 0;
                 try {
                      var allowedSpecial = new HashSet<string> { "<|endoftext|>" };
                      estimatedTotalStepsInEpoch = trainingData.Sum(s => Math.Max(0, tokenizer.Tokenize(s, allowedSpecial, applyPaddingTruncation: false).Length - 1));
                 } catch (Exception ex) {
                      Report($"Warning: Could not estimate total steps. Progress ETA might be inaccurate. Error: {ex.Message}");
                      estimatedTotalStepsInEpoch = -1; // Indica que a estimativa falhou
                 }


                try
                {
                    // --- Loop pelas Sentenças ---
                    foreach (string sentence in trainingData)
                    {
                        sentenceCount++;
                        int[] tokens;
                        try {
                            var allowedSpecialTokens = new HashSet<string> { "<|endoftext|>" };
                            // ***** MUDANÇA 2: Tokenizar SEM padding/truncamento *****
                            tokens = tokenizer.Tokenize(sentence, allowedSpecialTokens, applyPaddingTruncation: false);
                        } catch (Exception tex) {
                            Report($"ERROR tokenizing sentence {sentenceCount} (Epoch {epoch+1}): {tex.Message}. Skipping sentence.");
                            continue;
                        }

                        // Pula sequências muito curtas (sem tokens suficientes para formar par input/target)
                        if (tokens.Length <= 1) {
                            skippedSteps++; // Incrementa skip para sequência inteira
                            continue;
                        }

                        // --- Loop Interno pelos Passos (Tokens Alvo) ---
                        // Itera sobre todos os tokens REAIS da sequência, incluindo o EOS final como alvo
                        for (int i = 1; i < tokens.Length; i++)
                        {
                            long targetTokenId = tokens[i]; // Alvo é o token atual (pode ser EOS)

                            // ***** MUDANÇA 3: REMOVER o skip explícito do target *****
                            // if (targetTokenId == padTokenId) {
                            //    skippedSteps++;
                            //    continue;
                            // }
                            // ***** FIM MUDANÇA 3 *****


                            int[] inputSequenceTokens = tokens.Take(i).ToArray();
                            // Não deve mais acontecer com if (tokens.Length <= 1) acima, mas mantém por segurança
                            if (!inputSequenceTokens.Any()) { skippedSteps++; continue; }

                            Tensor? inputTensor = null, targetTensor = null, outputLogits = null, loss = null;

                            try // <- TRY INTERNO DO PASSO
                            {
                                inputTensor = tensor(inputSequenceTokens.Select(t => (long)t).ToArray(), dtype: ScalarType.Int64).to(device);
                                targetTensor = tensor(new long[] { targetTokenId }, dtype: ScalarType.Int64).to(device);

                                optimizer.zero_grad();
                                outputLogits = model.forward(inputTensor);

                                // Usa verificação de referência nula
                                if ((bool)(outputLogits == null)) { skippedSteps++; Report("Warning: Model returned null output."); continue; }

                                // Validação de Shape (importante)
                                if(outputLogits.dim() != 1 || outputLogits.shape[0] != vocabSize) {
                                    Report($"ERROR: Unexpected output shape {outputLogits.shape}. Expected [{vocabSize}]. Skipping step.");
                                    skippedSteps++; continue;
                                }

                                using var reshapedLogits = outputLogits.unsqueeze(0);
                                // Loss function sem ignore_index
                                loss = lossFunction.forward(reshapedLogits, targetTensor);
                                float currentLoss = loss.item<float>();

                                if (float.IsNaN(currentLoss) || float.IsInfinity(currentLoss)) { Report($"Warning: Invalid loss ({currentLoss}). Skipping step."); skippedSteps++; continue; }

                                loss.backward();
                                optimizer.step();

                                totalLoss += currentLoss;
                                totalStepsInEpoch++; // Passo bem-sucedido

                                // *** LOG DE PROGRESSO PERIÓDICO ***
                                if (totalStepsInEpoch > 0 && totalStepsInEpoch % reportIntervalSteps == 0 && estimatedTotalStepsInEpoch > 0)
                                {
                                    float avgLossSoFar = totalLoss / totalStepsInEpoch;
                                    double elapsedEpochMs = epochStopwatch.Elapsed.TotalMilliseconds;
                                    // Usa a estimativa de passos calculada no início da época
                                    double estimatedTotalEpochMs = (elapsedEpochMs / totalStepsInEpoch) * estimatedTotalStepsInEpoch;
                                    double estimatedRemainingMs = Math.Max(0, estimatedTotalEpochMs - elapsedEpochMs);
                                    TimeSpan remainingTs = TimeSpan.FromMilliseconds(estimatedRemainingMs);
                                    Report($"  Epoch {epoch + 1} Step {totalStepsInEpoch}/{estimatedTotalStepsInEpoch} [{DateTime.Now:HH:mm:ss}] - Avg Loss: {avgLossSoFar:F4} - Est. Epoch Rem: {remainingTs:hh\\:mm\\:ss}");
                                }
                            }
                            catch (Exception stepEx) { Report($"ERROR in training step (Sent# {sentenceCount}, Token# {i}, Epoch {epoch+1}): {stepEx.Message}"); skippedSteps++; }
                            finally { /* ... Dispose ... */ }
                        } // Fim loop interno (passos/tokens)
                    } // Fim loop externo (sentenças)
                }
                catch (Exception exOuterLoop) { Report($"ERROR in outer sentence loop (Epoch {epoch + 1}): {exOuterLoop.Message}"); }

                // --- Fim da Época ---
                model.eval();
                epochStopwatch.Stop();
                float avgLoss = totalStepsInEpoch > 0 ? totalLoss / totalStepsInEpoch : 0f;
                Report($"--- Epoch {epoch + 1} completed in {epochStopwatch.ElapsedMilliseconds} ms. Avg Loss: {avgLoss:F6}, Steps: {totalStepsInEpoch}, Skipped: {skippedSteps} ---");

            } // --- Fim do Loop Principal de Épocas ---

            totalStopwatch.Stop();
            model.eval();
            Report($"Training finished in {totalStopwatch.Elapsed}.");

            // --- Salvar Modelo ---
            Report($"Attempting to save final model state to: {this.modelSavePath}");
             try
            {
                 string? directory = Path.GetDirectoryName(this.modelSavePath);
                 if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory)) { Directory.CreateDirectory(directory); Report($"Created directory: {directory}"); }
                 this.model.save(this.modelSavePath);
                 Report($"Trained model state saved successfully to: {this.modelSavePath}");
                 if (!File.Exists(this.modelSavePath)) { Report("CRITICAL WARNING: Model file DOES NOT EXIST after save call!"); }
            }
            catch (Exception ex) { Report($"ERROR saving final model state: {ex.ToString()}"); }
        }
    } // --- Fim da Classe Trainer ---
} // --- Fim do Namespace ---