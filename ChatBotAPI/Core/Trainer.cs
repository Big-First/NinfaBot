// Trainer.cs - VERSÃO COM ESTRUTURA CORRETA (Save no final)

using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics; // Para Stopwatch
using System.IO; // Para Path, Directory, File
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.optim; // Para Adam
using static TorchSharp.torch.nn;    // Para CrossEntropyLoss, Module

namespace ChatBotAPI.Core
{
    public class Trainer
    {
        private readonly TorchSharpModel model;
        private readonly Tokenizer tokenizer;
        private readonly Optimizer optimizer;
        private readonly Module<Tensor, Tensor, Tensor> lossFunction; // Função de perda
        private readonly Device device; // Dispositivo (CPU ou CUDA)
        private readonly string modelSavePath; // Caminho para salvar

        // Construtor (Recebe save path, mas não TrainingState)
        public Trainer(TorchSharpModel model, Tokenizer tokenizer, double learningRate, string modelSavePath)
        {
            this.model = model ?? throw new ArgumentNullException(nameof(model));
            this.tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
            this.device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            this.modelSavePath = Path.GetFullPath(modelSavePath); // Garante caminho absoluto
            Console.WriteLine($"Trainer using device: {this.device.type}");
            Console.WriteLine($"Trainer configured to save model to: {this.modelSavePath}");
            this.model.to(this.device);
            this.optimizer = Adam(this.model.parameters(), lr: learningRate);
            int padTokenId = this.tokenizer.PadTokenId;
            this.lossFunction = CrossEntropyLoss(ignore_index: padTokenId).to(this.device);
            Console.WriteLine($"Trainer initialized loss function. Device: {this.device.type}. ignore_index (PadTokenId): {padTokenId}");
             if (tokenizer.ActualVocabSize <= 2) {
                 Console.Error.WriteLine("CRITICAL WARNING: Trainer detected Tokenizer ActualVocabSize <= 2...");
             }
        }

        // --- MÉTODO Train (Estrutura Corrigida) ---
        public void Train(List<string> trainingData, int epochs)
        {
            if (trainingData == null || !trainingData.Any()) { Console.WriteLine("Training data is empty. Skipping training."); return; }
            if (tokenizer.ActualVocabSize <= 0) { Console.Error.WriteLine("Cannot train: Tokenizer vocabulary size is invalid."); return; } // Verificação mais robusta

            int padTokenId = tokenizer.PadTokenId;
            int vocabSize = tokenizer.ActualVocabSize;
            int maxSeqLen = tokenizer.GetMaxSequenceLength(); // Usa o getter

            Console.WriteLine($"Starting TorchSharp training on {device.type}. Epochs: {epochs}. Sentences: {trainingData.Count}. PadTokenId: {padTokenId}. VocabSize: {vocabSize}. MaxSeqLen: {maxSeqLen}.");
            Stopwatch epochStopwatch = new Stopwatch();
            Stopwatch totalStopwatch = Stopwatch.StartNew(); // Tempo total

            // --- Loop Principal de Épocas ---
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                model.train(); // Modo treino no início da época
                epochStopwatch.Restart();
                Console.WriteLine($"--- Epoch {epoch + 1}/{epochs} ---");

                float totalLoss = 0f;
                long totalStepsInEpoch = 0;
                long skippedSteps = 0;
                long sentenceCount = 0;

                // *** Variável para controlar o log de progresso ***
                long reportIntervalSteps = 100; // Log a cada 100 passos de treino

                try
                {
                    // --- Loop pelas Sentenças ---
                    foreach (string sentence in trainingData)
                    {
                         sentenceCount++;
                         int[] tokens;
                         try {
                              tokens = tokenizer.Tokenize(sentence); // Chama o Tokenize da biblioteca agora
                         } catch (Exception tex) {
                             Console.Error.WriteLine($"ERROR tokenizing sentence {sentenceCount} (Epoch {epoch+1}): {tex.Message}. Skipping sentence.");
                             continue; // Pula esta sentença se a tokenização falhar
                         }


                         if (tokens.Length <= 1) {
                             skippedSteps += tokens.Length; // Conta como pulados
                             continue;
                         }

                         // --- Loop Interno pelos Passos (Tokens Alvo) ---
                         for (int i = 1; i < tokens.Length; i++)
                         {
                             long targetTokenId = tokens[i]; // O alvo é o token atual

                             // Pula se o *alvo* for padding
                             if (targetTokenId == padTokenId) {
                                 skippedSteps++;
                                 continue;
                             }

                             // A *entrada* são os tokens anteriores
                             // Não precisa mais converter para long aqui se Tokenize retorna int[]
                             int[] inputSequenceTokens = tokens.Take(i).ToArray();

                             // Pula se a sequência de entrada estiver vazia (não deve acontecer com i=1)
                             if (!inputSequenceTokens.Any()) { skippedSteps++; continue; }

                             Tensor? inputTensor = null;
                             Tensor? targetTensor = null;
                             Tensor? outputLogits = null;
                             Tensor? loss = null;

                             try // <- TRY INTERNO DO PASSO
                             {
                                 // Converte para LongTensor ANTES de enviar para o device
                                 inputTensor = tensor(inputSequenceTokens.Select(t => (long)t).ToArray(), dtype: ScalarType.Int64).to(device);
                                 // Target é um único valor, precisa ser um tensor 1D para CrossEntropyLoss
                                 targetTensor = tensor(new long[] { targetTokenId }, dtype: ScalarType.Int64).to(device);

                                 optimizer.zero_grad();
                                 outputLogits = model.forward(inputTensor);

                                 if ((bool)(outputLogits == null)) { skippedSteps++; continue; } // Verifica se a saída é válida

                                 // Verifica shape da saída (esperado: [vocabSize] ou [1, vocabSize])
                                 // CrossEntropyLoss espera [N, C] ou [C] para target [N] ou []
                                 // Nossa saída é [vocabSize], nosso target é [1]. Precisamos dar unsqueeze na saída.
                                 if(outputLogits.dim() != 1 || outputLogits.shape[0] != vocabSize) {
                                      Console.Error.WriteLine($"ERROR: Unexpected output shape {outputLogits.shape}. Expected [{vocabSize}]. Skipping step.");
                                      skippedSteps++;
                                      continue;
                                 }

                                 // Ajusta shape da saída para [1, vocabSize] para CrossEntropyLoss
                                 using var reshapedLogits = outputLogits.unsqueeze(0);

                                 // Calcular Loss, Backward, Step
                                 loss = lossFunction.forward(reshapedLogits, targetTensor); // Passa [1, C] e [1]
                                 float currentLoss = loss.item<float>();

                                 // Verifica se a perda é válida (NaN ou Infinito)
                                 if (float.IsNaN(currentLoss) || float.IsInfinity(currentLoss))
                                 {
                                     Console.Error.WriteLine($"Warning: Invalid loss detected ({currentLoss}) at step {totalStepsInEpoch + 1}. Skipping backward/step.");
                                     skippedSteps++;
                                     continue; // Pula backward e step
                                 }

                                 loss.backward();
                                 optimizer.step();

                                 // --- Atualiza contadores e loga progresso ---
                                 totalLoss += currentLoss;
                                 totalStepsInEpoch++; // Incrementa após um passo BEM SUCEDIDO

                                 // *** LOG DE PROGRESSO PERIÓDICO ***
                                 if (totalStepsInEpoch > 0 && totalStepsInEpoch % reportIntervalSteps == 0)
                                 {
                                     float avgLossSoFar = totalLoss / totalStepsInEpoch;
                                     // Calcula tempo estimado restante (simples)
                                     double elapsedEpochMs = epochStopwatch.Elapsed.TotalMilliseconds;
                                     double estimatedTotalEpochMs = (elapsedEpochMs / totalStepsInEpoch) * (trainingData.Sum(s=> Math.Max(0, s.Length-1))); // Estimativa grosseira baseada no total de tokens possíveis
                                     double estimatedRemainingMs = Math.Max(0, estimatedTotalEpochMs - elapsedEpochMs);
                                     TimeSpan remainingTs = TimeSpan.FromMilliseconds(estimatedRemainingMs);

                                     Console.WriteLine($"  Epoch {epoch + 1} Step {totalStepsInEpoch} [{DateTime.Now:HH:mm:ss}] - Avg Loss: {avgLossSoFar:F4} - Est. Epoch Rem: {remainingTs:hh\\:mm\\:ss}");
                                 }
                                 // *** FIM LOG DE PROGRESSO ***

                             }
                             catch (Exception stepEx) {
                                 Console.Error.WriteLine($"ERROR in training step (Sent# {sentenceCount}, Token# {i}, Epoch {epoch+1}): {stepEx.Message}");
                                 skippedSteps++;
                             }
                             finally {
                                // Dispose seguro dos tensores do passo
                                inputTensor?.Dispose();
                                targetTensor?.Dispose();
                                outputLogits?.Dispose();
                                loss?.Dispose();
                             }
                         } // Fim loop interno (passos/tokens)
                    } // Fim loop externo (sentenças)
                }
                catch (Exception exOuterLoop) { Console.Error.WriteLine($"ERROR in outer sentence loop (Epoch {epoch + 1}): {exOuterLoop.Message}"); }

                // --- Fim da Época ---
                model.eval(); // Modo avaliação no fim da época
                epochStopwatch.Stop();
                float avgLoss = totalStepsInEpoch > 0 ? totalLoss / totalStepsInEpoch : 0f;
                Console.WriteLine($"--- Epoch {epoch + 1} completed in {epochStopwatch.ElapsedMilliseconds} ms. Avg Loss: {avgLoss:F6}, Steps: {totalStepsInEpoch}, Skipped: {skippedSteps} ---");

            } // --- Fim do Loop Principal de Épocas ---

            totalStopwatch.Stop();
            model.eval(); // Garante modo de avaliação final
            Console.WriteLine($"Training finished in {totalStopwatch.Elapsed}.");

            // --- Salvar Modelo ---
            Console.WriteLine($"Attempting to save final model state to: {this.modelSavePath}");
            try
            {
                 // ... (código de salvar como antes) ...
                  string? directory = Path.GetDirectoryName(this.modelSavePath);
                  if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                  {
                      Directory.CreateDirectory(directory);
                      Console.WriteLine($"Created directory for model saving: {directory}");
                  }
                  this.model.save(this.modelSavePath);
                  Console.WriteLine($"Trained model state saved successfully to: {this.modelSavePath}");
                  if(!File.Exists(this.modelSavePath)) { Console.Error.WriteLine("CRITICAL WARNING: Model file DOES NOT EXIST after save call!"); }
            }
            catch (Exception ex) { Console.Error.WriteLine($"ERROR saving final model state: {ex.ToString()}"); }
        }
    } // --- Fim da Classe Trainer ---
} // --- Fim do Namespace ---