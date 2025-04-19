// Trainer.cs - AJUSTADO para aceitar TransformerModel e lógica de treino correta

using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks; // Adicionado para Task
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.nn;    // Garanta este using para Module, CrossEntropyLoss etc.

namespace ChatBotAPI.Core
{
    public class Trainer
    {
        // *** USA TIPO BASE COMPATÍVEL ***
        private readonly Module<Tensor, Tensor> model; // Aceita TransformerModel
        private readonly Tokenizer tokenizer;
        private readonly Optimizer optimizer;
        private readonly Module<Tensor, Tensor, Tensor> lossFunction;
        private readonly Device device;
        private readonly string modelSavePath;

        // *** CONSTRUTOR ACEITA Module<Tensor, Tensor> ***
        public Trainer(Module<Tensor, Tensor> model, Tokenizer tokenizer, double learningRate, string modelSavePath)
        {
            this.model = model ?? throw new ArgumentNullException(nameof(model));
            this.tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));

            // *** CORREÇÃO: Obtém o device do primeiro parâmetro do modelo ***
            var firstParam = this.model.parameters().FirstOrDefault();
            if ((bool)(firstParam == null))
            {
                // Se o modelo não tiver parâmetros (improvável, mas possível para modelos muito simples)
                // assume CPU ou lança erro. Vamos assumir CPU como fallback.
                Console.WriteLine("Warning: Model has no parameters. Assuming CPU device for Trainer.");
                this.device = torch.CPU;
                // Ou: throw new InvalidOperationException("Cannot determine model device: model has no parameters.");
            }
            else
            {
                this.device = firstParam.device; // Pega o device do primeiro parâmetro encontrado
            }
            this.modelSavePath = Path.GetFullPath(modelSavePath);
            Console.WriteLine($"Trainer using model on device: {this.device.type}"); // Loga o device real do modelo
            Console.WriteLine($"Trainer configured to save model to: {this.modelSavePath}");

            // Otimizador para os parâmetros do modelo (que já estão no device correto)
            this.optimizer = Adam(this.model.parameters(), lr: learningRate);
            int padTokenId = this.tokenizer.PadTokenId; // ID correto (50256)
            // Move a loss function para o mesmo device do modelo
            this.lossFunction = CrossEntropyLoss(ignore_index: padTokenId).to(this.device);
            Console.WriteLine($"Trainer initialized loss function. Device: {this.device.type}. ignore_index (PadTokenId): {padTokenId}");
            if (tokenizer.ActualVocabSize <= 0) { /* Log warning */ }
        }

        // --- MÉTODO Train AJUSTADO PARA TRANSFORMER ---
        public async Task Train(List<string> trainingData, int epochs) // Marcado como async para consistência
        {
            if (trainingData == null || !trainingData.Any()) { /*...*/ return; }
            if (tokenizer.ActualVocabSize <= 0) { /*...*/ return; }

            int padTokenId = tokenizer.PadTokenId;
            int vocabSize = tokenizer.ActualVocabSize;
            int maxSeqLen = tokenizer.GetMaxSequenceLength();

            Console.WriteLine($"Starting Transformer training on {device.type}. Epochs: {epochs}. Sequences: {trainingData.Count}. PadTokenId: {padTokenId}. VocabSize: {vocabSize}. MaxSeqLen: {maxSeqLen}.");
            Stopwatch epochStopwatch = new Stopwatch();
            Stopwatch totalStopwatch = Stopwatch.StartNew();

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                this.model.train(); // Coloca o Transformer em modo de treino (habilita dropout)
                epochStopwatch.Restart();
                Console.WriteLine($"--- Epoch {epoch + 1}/{epochs} ---");

                float totalLoss = 0f;
                long totalStepsInEpoch = 0; // Contará sequências processadas
                long skippedSequences = 0;
                long sentenceCount = 0;
                long reportIntervalSentences = 50; // Log a cada N sequências (ajuste conforme necessário)

                // Estimativa (opcional)
                long totalPossibleStepsEstimate = trainingData.Count;
                Console.WriteLine($"Processing {totalPossibleStepsEstimate} sequences per epoch.");


                try // Try da época
                {
                    foreach (string sentence in trainingData) // Formato: Input + Output<EOS>
                    {
                         sentenceCount++;
                         int[] tokens;
                         try {
                              // Tokeniza sequência completa. Padding/Truncamento é feito aqui.
                              tokens = tokenizer.Tokenize(sentence);
                         } catch (Exception tex) {
                             Console.Error.WriteLine($"ERROR tokenizing sentence {sentenceCount}: {tex.Message}. Skipping.");
                             skippedSequences++;
                             continue;
                         }

                         // Prepara input e target DESLOCADOS
                         // Input: Tira o último token (geralmente <EOS> ou PAD)
                         // Target: Tira o primeiro token
                         // Ambos terão comprimento SeqLen - 1 (ou menos se a original for curta)
                         int[] inputSequenceTokens = tokens.Take(tokens.Length - 1).ToArray();
                         int[] targetSequenceTokens = tokens.Skip(1).ToArray();

                         // Pula se algum ficou vazio após o deslocamento
                         if (inputSequenceTokens.Length == 0 || targetSequenceTokens.Length == 0) {
                              skippedSequences++;
                              continue;
                         }

                         // Garante que ambos tenham o mesmo comprimento (importante para loss)
                         // Se Tokenize já garante maxSeqLen, isso pode não ser estritamente necessário,
                         // mas é uma boa verificação.
                         if (inputSequenceTokens.Length != targetSequenceTokens.Length) {
                              Console.Error.WriteLine($"ERROR: Input ({inputSequenceTokens.Length}) and Target ({targetSequenceTokens.Length}) sequence lengths differ for sentence {sentenceCount}. Skipping.");
                              skippedSequences++;
                              continue;
                         }

                         Tensor? inputTensor = null;
                         Tensor? targetTensor = null;
                         Tensor? outputLogits = null; // Saída do Transformer: (SeqLen, Batch, VocabSize) ou (Batch, SeqLen, VocabSize)
                         Tensor? loss = null;
                         Tensor? reshapedLogits = null; // Para cálculo da loss
                         Tensor? reshapedTargets = null; // Para cálculo da loss


                         try // Try do passo de treino (uma sequência inteira)
                         {
                             // Cria tensores - Input shape (Batch=1, SeqLen)
                             inputTensor = tensor(inputSequenceTokens.Select(t => (long)t).ToArray(), dtype: ScalarType.Int64).unsqueeze(0).to(device);
                             // Target shape (SeqLen)
                             targetTensor = tensor(targetSequenceTokens.Select(t => (long)t).ToArray(), dtype: ScalarType.Int64).to(device);

                             optimizer.zero_grad();

                             // Forward pass - Espera (Batch, SeqLen) -> Retorna (SeqLen, Batch, VocabSize) [SE batch_first=False]
                             // *** GARANTA QUE TransformerModel.forward RETORNA LOGITS COMPLETOS ***
                             outputLogits = model.forward(inputTensor);

                             if ((bool)(outputLogits == null) || outputLogits.numel() == 0) {
                                 Console.Error.WriteLine($"ERROR: Model returned null or empty logits for sentence {sentenceCount}. Skipping step.");
                                 skippedSequences++; continue;
                             }
                             // Output esperado: (SeqLen_out, Batch=1, VocabSize) onde SeqLen_out = SeqLen_in

                             // --- Cálculo da Loss ---
                             // Logits precisam ser (N, C) e Targets (N)
                             // N = Batch * SeqLen = 1 * (SeqLen-1) = SeqLen-1
                             // C = vocabSize
                             reshapedLogits = outputLogits.view(-1, vocabSize); // Achata para (SeqLen-1, VocabSize)
                             reshapedTargets = targetTensor.view(-1);           // Achata para (SeqLen-1)

                             // Verifica shapes antes da loss
                             if (reshapedLogits.shape[0] != reshapedTargets.shape[0]) {
                                 Console.Error.WriteLine($"ERROR: Shape mismatch before loss! Logits: {reshapedLogits.shape}, Targets: {reshapedTargets.shape}. Skipping step.");
                                 skippedSequences++; continue;
                             }

                             loss = lossFunction.forward(reshapedLogits, reshapedTargets);
                             float currentLoss = loss.item<float>();

                             if (float.IsNaN(currentLoss) || float.IsInfinity(currentLoss)) { /*...*/ skippedSequences++; continue; }

                             loss.backward();
                             optimizer.step();

                             totalLoss += currentLoss;
                             totalStepsInEpoch++; // Conta sequência processada

                             // Log de Progresso
                             if (totalStepsInEpoch > 0 && sentenceCount % reportIntervalSentences == 0)
                             {
                                  float avgLossSoFar = totalLoss / totalStepsInEpoch;
                                  // ... (cálculo de tempo restante) ...
                                  Console.WriteLine($"  Epoch {epoch + 1} Sent# {sentenceCount}/{trainingData.Count} [...] - Avg Loss: {avgLossSoFar:F4} ...");
                                  // await SendDataToGraphAsync(...); // Se implementado
                             }
                         }
                         catch (Exception stepEx) { Console.Error.WriteLine($"ERROR training sentence {sentenceCount}: {stepEx}"); skippedSequences++; }
                         finally
                         {
                              // Dispose dos tensores do passo
                              inputTensor?.Dispose();
                              targetTensor?.Dispose();
                              outputLogits?.Dispose();
                              reshapedLogits?.Dispose(); // Descarta versões reshaped
                              reshapedTargets?.Dispose();
                              loss?.Dispose();
                         }

                    } // Fim loop foreach sentence
                }
                catch (Exception exOuterLoop) { Console.Error.WriteLine($"ERROR in outer epoch loop {epoch + 1}: {exOuterLoop}"); }

                // --- Fim da Época ---
                 model.eval(); // Modo de avaliação ao final da época
                 epochStopwatch.Stop();
                 float avgLoss = totalStepsInEpoch > 0 ? totalLoss / totalStepsInEpoch : 0f;
                 Console.WriteLine($"--- Epoch {epoch + 1} completed in {epochStopwatch.ElapsedMilliseconds} ms. Avg Loss: {avgLoss:F6}, Steps: {totalStepsInEpoch} (Sequences), Skipped: {skippedSequences} ---");

            } // --- Fim do Loop Principal de Épocas ---

            totalStopwatch.Stop();
            model.eval(); // Garante modo de avaliação final
            Console.WriteLine($"Training finished in {totalStopwatch.Elapsed}.");

            // --- Salvar Modelo ---
            Console.WriteLine($"Attempting to save final model state to: {this.modelSavePath}");
            try
            {
                 string? directory = Path.GetDirectoryName(this.modelSavePath);
                 if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory)) { Directory.CreateDirectory(directory); }
                 // Salva usando o método save do módulo (mais simples)
                 model.save(this.modelSavePath);
                 Console.WriteLine($"Trained model state saved successfully to: {this.modelSavePath}");
                 if(!File.Exists(this.modelSavePath)) { Console.Error.WriteLine("CRITICAL WARNING: Model file DOES NOT EXIST after save call!"); }
            }
            catch (Exception ex) { Console.Error.WriteLine($"ERROR saving final model state: {ex.ToString()}"); }
        } // --- Fim Método Train ---

    } // --- Fim Classe Trainer ---
} // --- Fim Namespace ---