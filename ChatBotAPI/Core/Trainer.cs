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
            // Verificações iniciais
            if (trainingData == null || !trainingData.Any()) { Console.WriteLine("Training data is empty. Skipping training."); return; }
            if (tokenizer.ActualVocabSize <= 2) { Console.Error.WriteLine("Cannot train: Tokenizer vocabulary is too small (Size <= 2)."); return; }

            int padTokenId = tokenizer.PadTokenId;
            int vocabSize = tokenizer.ActualVocabSize;
            int maxSeqLen = 50; // Valor padrão
             try { maxSeqLen = tokenizer.GetMaxSequenceLength(); }
             catch { Console.WriteLine($"Warning: Could not get MaxSequenceLength. Using fallback={maxSeqLen}."); }

            Console.WriteLine($"Starting TorchSharp training on {device.type}. Epochs: {epochs}. Sentences: {trainingData.Count}. PadTokenId: {padTokenId}. VocabSize: {vocabSize}. MaxSeqLen: {maxSeqLen}.");
            Stopwatch epochStopwatch = new Stopwatch();

            // --- Loop Principal de Épocas ---
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                // ***** model.train() NO INÍCIO DA ÉPOCA *****
                model.train();
                // ***** FIM model.train() *****

                epochStopwatch.Restart();
                Console.WriteLine($"--- Epoch {epoch + 1}/{epochs} ---");
                float totalLoss = 0f;
                long totalStepsInEpoch = 0;
                long skippedSteps = 0;
                long sentenceCount = 0;

                // Console.WriteLine($"DEBUG: Epoch {epoch + 1}: Entering sentence loop..."); // Log Opcional
                try
                {
                    // --- Loop pelas Sentenças ---
                    foreach (string sentence in trainingData)
                    {
                         sentenceCount++;
                         int[] tokens = tokenizer.Tokenize(sentence);

                         if (tokens.Length <= 1) {
                             // Console.WriteLine($"AVISO: Epoch {epoch + 1}, Sent {sentenceCount} -> tokens.Length <= 1. Skipping."); // Log Opcional
                             skippedSteps += tokens.Length;
                             continue;
                         }

                         // --- Loop Interno pelos Passos ---
                         for (int i = 1; i < tokens.Length; i++)
                         {
                             if (tokens[i] == padTokenId) {
                                 skippedSteps++;
                                 continue;
                             }

                             long[] inputSequence = tokens.Take(i).Select(t => (long)t).ToArray();
                             long targetTokenId = tokens[i];
                             if (!inputSequence.Any()) { skippedSteps++; continue; }

                             Tensor? inputTensor = null;
                             Tensor? targetTensor = null;
                             Tensor? outputLogits = null;
                             Tensor? loss = null;

                             try // <- TRY INTERNO DO PASSO
                             {
                                 inputTensor = tensor(inputSequence, dtype: ScalarType.Int64).to(device);
                                 targetTensor = tensor(new long[] { targetTokenId }, dtype: ScalarType.Int64).to(device);
                                 optimizer.zero_grad();
                                 outputLogits = model.forward(inputTensor);

                                 // Verificações (simplificadas, logs opcionais)
                                 if ((bool)(outputLogits == null)) { skippedSteps++; continue; }
                                 bool isShapeOk = (outputLogits.dim() == 1 && outputLogits.shape[0] == vocabSize);
                                 if (!isShapeOk) { skippedSteps++; continue; }

                                 // Calcular Loss, Backward, Step
                                 loss = lossFunction.forward(outputLogits!.unsqueeze(0), targetTensor!);
                                 float currentLoss = loss!.item<float>();
                                 loss.backward();
                                 optimizer.step();
                                 totalLoss += currentLoss;
                                 totalStepsInEpoch++;

                             }
                             catch (Exception stepEx) {
                                 Console.Error.WriteLine($"ERROR in training step (i={i}, sent={sentenceCount}, epoch={epoch+1}): {stepEx.Message}");
                                 skippedSteps++;
                             }
                             finally {
                                inputTensor?.Dispose(); targetTensor?.Dispose(); outputLogits?.Dispose(); loss?.Dispose();
                             }
                         } // Fim loop interno (passos)
                    } // Fim loop externo (sentenças)
                    // Console.WriteLine($"DEBUG: Epoch {epoch + 1}: Finished sentence loop."); // Log Opcional
                }
                catch (Exception exOuterLoop) { Console.Error.WriteLine($"ERROR in outer sentence loop (Epoch {epoch + 1}): {exOuterLoop.Message}"); }

                // --- Fim da Época ---
                model.eval(); // Modo avaliação no fim da época
                epochStopwatch.Stop();
                float avgLoss = totalStepsInEpoch > 0 ? totalLoss / totalStepsInEpoch : 0f;
                Console.WriteLine($"--- Epoch {epoch + 1} completed in {epochStopwatch.ElapsedMilliseconds} ms. Avg Loss: {avgLoss:F6}, Steps: {totalStepsInEpoch}, Skipped: {skippedSteps} ---");

            } // --- Fim do Loop Principal de Épocas ---

            // ***** model.eval() FINAL *****
            model.eval();
            Console.WriteLine("Training finished.");

            // ***** SALVAR O MODELO APENAS NO FINAL DO TREINO COMPLETO *****
            Console.WriteLine($"Attempting to save final model state to: {this.modelSavePath}");
            try
            {
                 string? directory = Path.GetDirectoryName(this.modelSavePath);
                 if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                 {
                     Directory.CreateDirectory(directory);
                     Console.WriteLine($"Created directory for model saving: {directory}");
                 }
                 this.model.save(this.modelSavePath); // Salva o estado final
                 Console.WriteLine($"Trained model state saved successfully to: {this.modelSavePath}");
                 if(!File.Exists(this.modelSavePath)) { Console.Error.WriteLine("CRITICAL WARNING: Model file DOES NOT EXIST after save call!"); }
            }
            catch (Exception ex) { Console.Error.WriteLine($"ERROR saving final model state: {ex.ToString()}"); }
            // ***** FIM SALVAR MODELO *****

        } // --- Fim do Método Train ---
    } // --- Fim da Classe Trainer ---
} // --- Fim do Namespace ---