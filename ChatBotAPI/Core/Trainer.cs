 // Trainer.cs - Com Weight Decay e Early Stopping implementados

 using System;
 using System.Collections.Generic;
 using System.Linq;
 using System.Diagnostics;
 using System.IO;
 using System.Threading.Tasks;
 using TorchSharp;
 using static TorchSharp.torch;
 using static TorchSharp.torch.optim;
 using static TorchSharp.torch.nn;

 namespace ChatBotAPI.Core
 {
     public class Trainer
     {
         private readonly Module<Tensor, Tensor> model; // Aceita TransformerModel
         private readonly Tokenizer tokenizer;
         private readonly Optimizer optimizer;
         private readonly Module<Tensor, Tensor, Tensor> lossFunction;
         private readonly Device device;
         private readonly string modelSavePath;

         // Campos para Early Stopping
         private double bestValidationLoss = double.PositiveInfinity;
         private int epochsSinceImprovement = 0;

         // Construtor aceita weightDecay
         public Trainer(Module<Tensor, Tensor> model, Tokenizer tokenizer, double learningRate, string modelSavePath, double weightDecay = 1e-5) // Adicionado weightDecay
         {
             this.model = model ?? throw new ArgumentNullException(nameof(model));
             this.tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));

             var firstParam = this.model.parameters().FirstOrDefault();
             if ((bool)(firstParam == null)) {
                  Console.WriteLine("Warning: Model has no parameters. Assuming CPU device for Trainer.");
                  this.device = torch.CPU;
             } else { this.device = firstParam.device; }

             this.modelSavePath = Path.GetFullPath(modelSavePath);
             Console.WriteLine($"Trainer using model on device: {this.device.type}");
             Console.WriteLine($"Trainer configured to save model to: {this.modelSavePath}");

             // *** ADICIONA weight_decay ao Adam ***
             this.optimizer = Adam(this.model.parameters(), lr: learningRate, weight_decay: weightDecay);
             Console.WriteLine($"Trainer initialized Adam optimizer with LR={learningRate}, WeightDecay={weightDecay}");

             int padTokenId = this.tokenizer.PadTokenId;
             this.lossFunction = CrossEntropyLoss(ignore_index: padTokenId).to(this.device);
             Console.WriteLine($"Trainer initialized loss function. Device: {this.device.type}. ignore_index (PadTokenId): {padTokenId}");
             if (tokenizer.ActualVocabSize <= 0) { /* Log warning */ }
         }

         // --- NOVO MÉTODO Evaluate (privado) ---
         private double Evaluate(List<string> validationData)
         {
             if (validationData == null || !validationData.Any()) return double.PositiveInfinity;

             this.model.eval(); // MODO DE AVALIAÇÃO
             double totalLoss = 0;
             long totalSteps = 0;
             int vocabSize = tokenizer.ActualVocabSize;
             int maxSeqLen = tokenizer.GetMaxSequenceLength(); // Pega o maxSeqLen

             using (var noGrad = torch.no_grad())
             {
                 foreach (string sentence in validationData)
                 {
                     List<int> actualTokens; // Tokens reais
                     try {
                          actualTokens = tokenizer.GetTokenIdsWithoutPadding(sentence);
                          if (actualTokens.Count > maxSeqLen) actualTokens = actualTokens.GetRange(0, maxSeqLen);
                     } catch { continue; }

                     if (actualTokens.Count <= 1) continue;

                     int[] inputSequenceTokens = actualTokens.Take(actualTokens.Count - 1).ToArray();
                     int[] targetSequenceTokens = actualTokens.Skip(1).ToArray();
                     if (inputSequenceTokens.Length == 0) continue; // Sanity check

                     Tensor? inputTensor = null;
                     Tensor? targetTensor = null;
                     Tensor? outputLogits = null;
                     Tensor? loss = null;
                     Tensor? reshapedLogits = null;
                     Tensor? reshapedTargets = null;

                     try
                     {
                         inputTensor = tensor(inputSequenceTokens.Select(t => (long)t).ToArray(), dtype: ScalarType.Int64).unsqueeze(0).to(device);
                         targetTensor = tensor(targetSequenceTokens.Select(t => (long)t).ToArray(), dtype: ScalarType.Int64).to(device);

                         outputLogits = model.forward(inputTensor); // Assume retorno de sequência completa
                         if ((bool)(outputLogits == null) || outputLogits.numel() == 0) continue;

                         reshapedLogits = outputLogits.view(-1, vocabSize);
                         reshapedTargets = targetTensor.view(-1);
                         if (reshapedLogits.shape[0] != reshapedTargets.shape[0]) continue;

                         loss = lossFunction.forward(reshapedLogits, reshapedTargets);
                         float currentLoss = loss.item<float>();

                         if (!float.IsNaN(currentLoss) && !float.IsInfinity(currentLoss))
                         {
                             totalLoss += currentLoss;
                             totalSteps++;
                         }
                     }
                     catch (Exception ex) { Console.Error.WriteLine($"Error during validation sentence: {ex.Message}"); }
                     finally { /* Dispose dos tensores locais do Evaluate */
                          inputTensor?.Dispose(); targetTensor?.Dispose(); outputLogits?.Dispose();
                          reshapedLogits?.Dispose(); reshapedTargets?.Dispose(); loss?.Dispose();
                     }
                 }
             }
             this.model.train(); // Volta para modo de treino após avaliar
             return totalSteps > 0 ? totalLoss / totalSteps : double.PositiveInfinity;
         }


         // --- MÉTODO Train MODIFICADO para Early Stopping ---
         // Aceita validationData e patience
         public async Task Train(List<string> trainingData, List<string> validationData, int epochs, int patience = 3)
         {
            if (trainingData == null || !trainingData.Any()) { /*...*/ return; }
            if (tokenizer.ActualVocabSize <= 0) { /*...*/ return; }

            int padTokenId = tokenizer.PadTokenId;
            int vocabSize = tokenizer.ActualVocabSize;
            int maxSeqLen = tokenizer.GetMaxSequenceLength();

            Console.WriteLine($"Starting Transformer training on {device.type}. Epochs: {epochs}. Patience: {patience}. Train Seq: {trainingData.Count}. Val Seq: {validationData?.Count ?? 0}.");
            Stopwatch epochStopwatch = new Stopwatch();
            Stopwatch totalStopwatch = Stopwatch.StartNew();

            // Reseta estado do Early Stopping
            bestValidationLoss = double.PositiveInfinity;
            epochsSinceImprovement = 0;
            string bestModelPath = Path.ChangeExtension(modelSavePath, ".best.pt"); // Define caminho do melhor modelo

            try // Bloco try principal para garantir que StopGraphProcess seja chamado
            {
                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    this.model.train(); // Garante modo treino no início da época
                    epochStopwatch.Restart();
                    Console.WriteLine($"--- Epoch {epoch + 1}/{epochs} ---");

                    float totalTrainLoss = 0f; // Renomeado para clareza
                    long totalTrainSteps = 0;  // Renomeado para clareza
                    long skippedSequences = 0;
                    long sentenceCount = 0;
                    long reportIntervalSentences = 50;

                    // ... (Estimativa de passos - opcional) ...

                    try // Try da época de treino
                    {
                        foreach (string sentence in trainingData)
                        {
                             sentenceCount++;
                             List<int> actualTokens;
                             try {
                                  actualTokens = tokenizer.GetTokenIdsWithoutPadding(sentence);
                                  if (actualTokens.Count > maxSeqLen) actualTokens = actualTokens.GetRange(0, maxSeqLen);
                             } catch (Exception tex) { /*...*/ skippedSequences++; continue; }

                             if (actualTokens.Count <= 1) { skippedSequences++; continue; }

                             int[] inputSequenceTokens = actualTokens.Take(actualTokens.Count - 1).ToArray();
                             int[] targetSequenceTokens = actualTokens.Skip(1).ToArray();
                             if (inputSequenceTokens.Length == 0) { skippedSequences++; continue; } // Verifica input vazio
                             if (inputSequenceTokens.Length != targetSequenceTokens.Length) { /*...*/ skippedSequences++; continue; } // Verifica comprimento

                             Tensor? inputTensor = null, targetTensor = null, outputLogits = null, loss = null;
                             Tensor? reshapedLogits = null, reshapedTargets = null;

                             try // Try do passo de treino
                             {
                                 inputTensor = tensor(inputSequenceTokens.Select(t => (long)t).ToArray(), dtype: ScalarType.Int64).unsqueeze(0).to(device);
                                 targetTensor = tensor(targetSequenceTokens.Select(t => (long)t).ToArray(), dtype: ScalarType.Int64).to(device);
                                 optimizer.zero_grad();
                                 outputLogits = model.forward(inputTensor); // Assume retorno de sequência completa
                                 if ((bool)(outputLogits == null) || outputLogits.numel() == 0) { /*...*/ skippedSequences++; continue; }

                                 reshapedLogits = outputLogits.view(-1, vocabSize);
                                 reshapedTargets = targetTensor.view(-1);
                                 if (reshapedLogits.shape[0] != reshapedTargets.shape[0]) { /*...*/ skippedSequences++; continue; }

                                 loss = lossFunction.forward(reshapedLogits, reshapedTargets);
                                 float currentLoss = loss.item<float>();
                                 if (float.IsNaN(currentLoss) || float.IsInfinity(currentLoss)) { /*...*/ skippedSequences++; continue; }

                                 loss.backward();
                                 optimizer.step();

                                 totalTrainLoss += currentLoss; // Acumula perda de treino
                                 totalTrainSteps++;

                                 // Log de Progresso
                                 if (totalTrainSteps > 0 && sentenceCount % reportIntervalSentences == 0) { /*...*/ }
                             }
                             catch (Exception stepEx) { /*...*/ skippedSequences++; }
                             finally { /* ... dispose tensores do passo ... */ }

                        } // Fim loop foreach sentence
                    } // Fim Try da época de treino
                    catch (Exception exOuterLoop) { /*...*/ }

                    // --- AVALIAÇÃO E EARLY STOPPING AO FINAL DA ÉPOCA ---
                    double currentValidationLoss = Evaluate(validationData); // Avalia no conjunto de validação
                    epochStopwatch.Stop();

                     float avgTrainLoss = totalTrainSteps > 0 ? totalTrainLoss / totalTrainSteps : float.NaN;
                     Console.WriteLine($"--- Epoch {epoch + 1} completed in {epochStopwatch.ElapsedMilliseconds} ms. Avg Train Loss: {avgTrainLoss:F6}, Avg Validation Loss: {(double.IsPositiveInfinity(currentValidationLoss) ? "N/A" : currentValidationLoss.ToString("F6"))}, Steps: {totalTrainSteps}, Skipped: {skippedSequences} ---");

                    // Lógica de Early Stopping
                    if (validationData != null && validationData.Any()) // Só faz early stopping se tiver dados de validação
                    {
                         if (currentValidationLoss < bestValidationLoss)
                         {
                             Console.WriteLine($"Validation loss improved ({bestValidationLoss:F6} --> {currentValidationLoss:F6}). Saving best model...");
                             bestValidationLoss = currentValidationLoss;
                             epochsSinceImprovement = 0;
                             try { // Salva o melhor modelo
                                 string? dir = Path.GetDirectoryName(bestModelPath);
                                 if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir)) Directory.CreateDirectory(dir);
                                 model.save(bestModelPath); // Salva o modelo atual como o melhor
                                 Console.WriteLine($"Best model state saved to: {bestModelPath}");
                             } catch (Exception ex) { Console.Error.WriteLine($"ERROR saving best model state: {ex}"); }
                         }
                         else
                         {
                             epochsSinceImprovement++;
                             Console.WriteLine($"Validation loss did not improve for {epochsSinceImprovement} epoch(s). Best was {bestValidationLoss:F6}.");
                             if (epochsSinceImprovement >= patience)
                             {
                                 Console.WriteLine($"--- Early stopping triggered after {patience} epochs without improvement. ---");
                                 break; // Interrompe o loop FOR das épocas
                             }
                         }
                    } else {
                         // Se não há dados de validação, salva o modelo da última época (opcional)
                         // Considerar salvar a cada N épocas?
                    }
                     // --- FIM AVALIAÇÃO E EARLY STOPPING ---

                } // --- Fim do Loop FOR Principal de Épocas ---
            } // Fim Try Principal
            finally
            {
                 // Código para parar o gráfico se implementado
                 // StopGraphProcess();
            }


            totalStopwatch.Stop();
            model.eval(); // Garante modo de avaliação final
            Console.WriteLine($"Training finished in {totalStopwatch.Elapsed}. Best Validation Loss: {(double.IsPositiveInfinity(bestValidationLoss) ? "N/A" : bestValidationLoss.ToString("F6"))}");

            // DECISÃO: Salvar o último modelo ou apenas o melhor?
            // Se quiser usar o MELHOR modelo (recomendado), carregue-o aqui antes de sair,
            // ou informe o usuário para usar o arquivo .best.pt
            if (File.Exists(bestModelPath)) {
                 Console.WriteLine($"Final model state will be based on the best model saved: {bestModelPath}");
                 // Opcional: Carregar o melhor estado no objeto 'model' atual
                 // try { model.load_state_dict(torch.load(bestModelPath, map_location: this.device)); } catch {}
            } else {
                 Console.WriteLine("No best model saved (no validation data or no improvement). Using model from last epoch.");
                  // Salva o modelo da última época se não houver 'best'
                  Console.WriteLine($"Attempting to save final model state (last epoch) to: {this.modelSavePath}");
                  try
                  {
                      model.save(this.modelSavePath); 
                  } catch(Exception ex) {
                      Console.WriteLine(ex.Message);
                  }
            }
        } // --- Fim Método Train ---

    } // --- Fim Classe Trainer ---
} // --- Fim Namespace ---