using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using AI.Core;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.nn;

namespace AI.Utils
{
    /// <summary>
    /// Handles the training process for a Transformer neural network model using TorchSharp.
    /// This class manages model training with batching, padding, and loss calculation.
    /// </summary>
    public class TransformerTrainer
    {
        /// <summary>
        /// The neural network model being trained (espera-se que seja um TransformerModel ou compatível).
        /// </summary>
        private readonly Module<Tensor, Tensor> model; // Usa a base Module para flexibilidade

        /// <summary>
        /// Tokenizer used for converting text into token sequences.
        /// </summary>
        private readonly Tokenizer tokenizer;

        /// <summary>
        /// Optimizer used for updating model parameters during training.
        /// </summary>
        private readonly Optimizer optimizer;

        /// <summary>
        /// Loss function used to calculate training loss, ignoring padding.
        /// </summary>
        private readonly Module<Tensor, Tensor, Tensor> lossFunction;

        /// <summary>
        /// The device (CPU/GPU) where the model is being trained.
        /// </summary>
        private readonly Device device;

        /// <summary>
        /// Path where the trained model will be saved.
        /// </summary>
        private readonly string modelSavePath;

        /// <summary>
        /// Size of the batches used during training.
        /// </summary>
        private readonly int batchSize;

        /// <summary>
        /// Maximum sequence length for padding and model input.
        /// </summary>
        private readonly int maxSequenceLength;

        /// <summary>
        /// Action delegate for reporting training progress.
        /// </summary>
        private Action<string> Report = Console.WriteLine;

        /// <summary>
        /// Initializes a new instance of the TransformerTrainer class.
        /// </summary>
        /// <param name="model">The neural network model to train.</param>
        /// <param name="tokenizer">Tokenizer for converting text to tokens.</param>
        /// <param name="learningRate">Learning rate for the optimizer.</param>
        /// <param name="modelSavePath">Path where the trained model will be saved.</param>
        /// <param name="batchSize">Batch size for training.</param>
        /// <exception cref="ArgumentNullException">Thrown when model or tokenizer is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when batchSize is invalid.</exception>
        public TransformerTrainer(
            Module<Tensor, Tensor> model, // Aceita Module<Tensor, Tensor> se TransformerModel herdar disso
            Tokenizer tokenizer,
            double learningRate,
            string modelSavePath,
            int batchSize = 8) // Tamanho de batch padrão
        {
            this.model = model ?? throw new ArgumentNullException(nameof(model));
            this.tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
            if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be positive.");
            this.batchSize = batchSize;
            this.maxSequenceLength = tokenizer.GetMaxSequenceLength(); // Obtenha o max_seq_len do tokenizer

            this.device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            this.modelSavePath = Path.GetFullPath(modelSavePath);
            Console.WriteLine($"TransformerTrainer using device: {this.device.type}");
            Console.WriteLine($"TransformerTrainer configured with Batch Size: {this.batchSize}, MaxSeqLen: {this.maxSequenceLength}");
            Console.WriteLine($"TransformerTrainer configured to save model to: {this.modelSavePath}");

            this.model.to(this.device);

            // Usar AdamW é comum para Transformers (inclui weight decay)
            this.optimizer = AdamW(this.model.parameters(), lr: learningRate /*, weight_decay: 0.01 */);

            // ***** Configurar CrossEntropyLoss para IGNORAR o PadTokenId *****
            // Isso é CRUCIAL porque estamos adicionando padding, e NÃO queremos
            // que o modelo seja penalizado por não prever o token de padding
            // nas posições de padding.
            long padTokenIdLong = tokenizer.PadTokenId;
            this.lossFunction = CrossEntropyLoss(ignore_index: padTokenIdLong).to(this.device);
            Console.WriteLine($"TransformerTrainer initialized loss function (CrossEntropyLoss) with ignore_index: {padTokenIdLong} (PadTokenId). Device: {this.device.type}.");

            if (tokenizer.ActualVocabSize <= 2) { /* ... alerta ... */ }
             if (this.maxSequenceLength <= 1) { Report("Warning: MaxSequenceLength from tokenizer is too short for training."); }
        }

        /// <summary>
        /// Trains the model on the provided training data in batches.
        /// </summary>
        /// <param name="trainingData">List of raw training sentences/sequences.</param>
        /// <param name="epochs">Number of training epochs.</param>
        /// <param name="progressReporter">Optional progress reporter.</param>
        public void Train(List<string> trainingData, int epochs, IProgress<string>? progressReporter = null)
        {
             if (progressReporter != null) {
                 Report = message => { Console.WriteLine(message); progressReporter.Report(message); };
             } else {
                  Report = Console.WriteLine; // Garante que Report não seja nulo
             }

            if (trainingData == null || !trainingData.Any()) { Report("Training data is empty. Skipping training."); return; }
            if (tokenizer.ActualVocabSize <= 0) { Report("ERROR: Cannot train: Tokenizer vocabulary size is invalid."); return; }
             if (this.maxSequenceLength <= 1) { Report("ERROR: Cannot train: MaxSequenceLength is too short."); return; }

            Report($"Starting Transformer training on {device.type}. Epochs: {epochs}. Sentences: {trainingData.Count}. Batch Size: {batchSize}. MaxSeqLen: {maxSequenceLength}.");
            Stopwatch epochStopwatch = new Stopwatch();
            Stopwatch totalStopwatch = Stopwatch.StartNew();

            try
            {
                // *** 1. Tokenizar e Preparar Dados em Batches ***
                // Tokenize todas as sequências primeiro
                List<int[]> tokenizedSequences = trainingData
                     .Select(s => tokenizer.Tokenize(s, applyPaddingTruncation: false)) // NÃO aplica padding/truncamento aqui
                     .Where(tokens => tokens.Length > 1) // Pula sequências muito curtas
                     .ToList();

                Report($"Prepared {tokenizedSequences.Count} tokenized sequences (min length 2).");

                // Crie batches de tensores (input e target) com padding
                var dataBatches = CreatePaddedBatches(tokenizedSequences, this.batchSize, this.maxSequenceLength, this.tokenizer.PadTokenId);
                int numberOfBatches = dataBatches.Count;

                Report($"Created {numberOfBatches} batches for training.");

                if (numberOfBatches == 0) { Report("No valid data batches created. Skipping training."); return; }

                // --- Loop Principal de Épocas ---
                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    model.train(); // Coloca o modelo em modo de treinamento (dropout ativo, etc.)
                    epochStopwatch.Restart();
                    Report($"--- Epoch {epoch + 1}/{epochs} ---");

                    float totalLoss = 0f;
                    long totalBatchesInEpoch = 0;

                    try
                    {
                        // --- Loop pelos Batches ---
                        foreach (var batch in dataBatches)
                        {
                            // batch.Item1: input tensor (Batch, SeqLen)
                            // batch.Item2: target tensor (Batch, SeqLen)

                            Tensor? inputTensor = null;
                            Tensor? targetTensor = null;
                            Tensor? outputLogits = null;
                            Tensor? loss = null;

                            try // <- TRY INTERNO DO BATCH
                            {
                                inputTensor = batch.Item1.to(device); // Move o batch de input para o dispositivo
                                targetTensor = batch.Item2.to(device); // Move o batch de target para o dispositivo

                                optimizer.zero_grad(); // Zera os gradientes acumulados

                                // *** Forward Pass ***
                                // O modelo TransformerModel (Decoder-Only) espera input (Batch, SeqLen)
                                // E retorna logits (Batch, SeqLen, VocabSize)
                                // Ele lida com a máscara causal internamente.
                                outputLogits = model.forward(inputTensor);

                                // Verifica nulidade de outputLogits
                                if ((bool)(outputLogits == null)) { Report("Warning: Model returned null output logits for batch. Skipping batch."); continue; }

                                // Validação de Shape de Saída (importante)
                                // Shape esperado: (BatchSize do batch atual, MaxSeqLen ou padded_seq_len, VocabSize)
                                var expectedLogitsShape = new long[] { inputTensor.shape[0], inputTensor.shape[1], tokenizer.ActualVocabSize };
                                if (!outputLogits.shape.SequenceEqual(expectedLogitsShape)) {
                                    Report($"ERROR: Unexpected output shape {outputLogits.shape} for batch. Expected {expectedLogitsShape}. Skipping batch.");
                                    continue;
                                }

                                // *** Cálculo da Loss ***
                                // CrossEntropyLoss espera logits (N, C, D1, D2...) e targets (N, D1, D2...)
                                // N = BatchSize, C = VocabSize.
                                // Nossos logits são (Batch, SeqLen, VocabSize). Nossos targets são (Batch, SeqLen).
                                // Precisamos remodelar os logits para (Batch * SeqLen, VocabSize) e targets para (Batch * SeqLen)
                                // para usar a função de perda simples com ignore_index.

                                // Remodela Logits: (Batch * SeqLen, VocabSize)
                                using var reshapedLogits = outputLogits.view(-1, tokenizer.ActualVocabSize);
                                // Remodela Targets: (Batch * SeqLen)
                                using var reshapedTargets = targetTensor.view(-1);

                                // Calcula a perda. A perda ignora os alvos que são o PadTokenId.
                                loss = lossFunction.forward(reshapedLogits, reshapedTargets);
                                float currentLoss = loss.item<float>();

                                // Verifica se a perda é válida
                                if (float.IsNaN(currentLoss) || float.IsInfinity(currentLoss)) { Report($"Warning: Invalid loss ({currentLoss}). Skipping batch."); continue; }

                                // *** Backpropagation e Otimização ***
                                loss.backward(); // Calcula os gradientes
                                optimizer.step(); // Atualiza os pesos do modelo

                                totalLoss += currentLoss;
                                totalBatchesInEpoch++; // Passo bem-sucedido

                                // *** LOG DE PROGRESSO PERIÓDICO ***
                                if (totalBatchesInEpoch % Math.Max(1, numberOfBatches / 10) == 0) // Log 10 vezes por época
                                {
                                    float avgLossSoFar = totalLoss / totalBatchesInEpoch;
                                    double elapsedEpochMs = epochStopwatch.Elapsed.TotalMilliseconds;
                                    double estimatedTotalEpochMs = (elapsedEpochMs / totalBatchesInEpoch) * numberOfBatches;
                                    double estimatedRemainingMs = Math.Max(0, estimatedTotalEpochMs - elapsedEpochMs);
                                    TimeSpan remainingTs = TimeSpan.FromMilliseconds(estimatedRemainingMs);
                                    Report($"  Epoch {epoch + 1} Batch {totalBatchesInEpoch}/{numberOfBatches} [{DateTime.Now:HH:mm:ss}] - Avg Loss: {avgLossSoFar:F4} - Est. Epoch Rem: {remainingTs:hh\\:mm\\:ss}");
                                }
                            }
                            catch (Exception batchEx) { Report($"ERROR in training batch {totalBatchesInEpoch + 1} (Epoch {epoch+1}): {batchEx.Message}"); }
                            finally {
                                // Dispose dos tensores criados dentro do try do batch
                                inputTensor?.Dispose();
                                targetTensor?.Dispose();
                                outputLogits?.Dispose();
                                loss?.Dispose();
                                reshapedLogits?.Dispose(); // Descartar tensores remodados temporários
                                reshapedTargets?.Dispose();
                            }
                        } // Fim loop pelos batches
                    }
                    catch (Exception exOuterLoop) { Report($"ERROR in outer batch loop (Epoch {epoch + 1}): {exOuterLoop.Message}"); }

                    // --- Fim da Época ---
                    model.eval(); // Coloca o modelo em modo de avaliação
                    epochStopwatch.Stop();
                    float avgLoss = totalBatchesInEpoch > 0 ? totalLoss / totalBatchesInEpoch : 0f;
                    Report($"--- Epoch {epoch + 1} completed in {epochStopwatch.ElapsedMilliseconds} ms. Avg Loss: {avgLoss:F6}, Batches: {totalBatchesInEpoch} ---");

                } // --- Fim do Loop Principal de Épocas ---
            }
            catch (Exception trainEx)
            {
                Report($"ERROR during training process: {trainEx.ToString()}");
            }
            finally
            {
                 totalStopwatch.Stop();
                 model.eval(); // Garante modo de avaliação no final
                 Report($"Training finished in {totalStopwatch.Elapsed}.");

                 // --- Salvar Modelo ---
                 Report($"Attempting to save final model state to: {this.modelSavePath}");
                  try
                 {
                      string? directory = Path.GetDirectoryName(this.modelSavePath);
                      if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory)) { Directory.CreateDirectory(directory); Report($"Created directory: {directory}"); }
                      this.model.save(this.modelSavePath); // model.save() funciona porque Module<Tensor, Tensor> tem o método save
                      Report($"Trained model state saved successfully to: {this.modelSavePath}");
                      if (!File.Exists(this.modelSavePath)) { Report("CRITICAL WARNING: Model file DOES NOT EXIST after save call!"); }
                 }
                 catch (Exception ex) { Report($"ERROR saving final model state: {ex.ToString()}"); }
            }
        } // --- Fim da Função Train ---


        /// <summary>
        /// Helper method to create padded batches of tokenized sequences.
        /// </summary>
        /// <param name="tokenizedSequences">List of tokenized sequences (int arrays).</param>
        /// <param name="batchSize">Size of each batch.</param>
        /// <param name="maxSequenceLength">Maximum sequence length for padding.</param>
        /// <param name="padTokenId">The token ID to use for padding.</param>
        /// <returns>A list of tuples, each containing the input tensor and target tensor for a batch.</returns>
        private List<Tuple<Tensor, Tensor>> CreatePaddedBatches(
            List<int[]> tokenizedSequences,
            int batchSize,
            int maxSequenceLength,
            int padTokenId)
        {
            var batches = new List<Tuple<Tensor, Tensor>>();
            // Garante embaralhamento para que o modelo não veja a mesma ordem de dados a cada época
            var shuffledSequences = tokenizedSequences.OrderBy(_ => Guid.NewGuid()).ToList();

            for (int i = 0; i < shuffledSequences.Count; i += batchSize)
            {
                // Pega um batch de sequências
                var batchSequences = shuffledSequences.Skip(i).Take(batchSize).ToList();

                // Encontra o comprimento máximo da sequência NESTE batch
                // Ou usa o maxSequenceLength global se batches devem ser de tamanho fixo
                // int currentBatchMaxLen = batchSequences.Max(s => s.Length);
                // currentBatchMaxLen = Math.Min(currentBatchMaxLen, maxSequenceLength); // Limita ao global max

                // Para treinamento de LM (Language Model), é comum padronizar todos os batches
                // para o maxSequenceLength do modelo, para shapes consistentes.
                int currentBatchTargetLen = maxSequenceLength; // O comprimento do input e target após padding/truncamento

                // Listas temporárias para construir o batch
                var inputBatchList = new List<long[]>();
                var targetBatchList = new List<long[]>();

                foreach (var seq in batchSequences)
                {
                    // Trunca se a sequência for mais longa que o max global (isso DEVE acontecer antes)
                    var processedSeq = seq.Length > maxSequenceLength
                        ? seq.Take(maxSequenceLength).ToArray()
                        : seq;

                    // Pula sequências que, mesmo truncadas, são muito curtas para formar um par input/target válido
                     if (processedSeq.Length <= 1) {
                         // Console.WriteLine($"Warning: Skipping sequence of length {processedSeq.Length} in batch creation.");
                         continue; // Pula para a próxima sequência no batch
                     }

                    // Cria a sequência de input (todos os tokens exceto o último)
                    var inputSeq = processedSeq.Take(processedSeq.Length - 1).ToList();
                    // Cria a sequência de target (todos os tokens exceto o primeiro)
                    var targetSeq = processedSeq.Skip(1).ToList();

                    // --- Padding ---
                    // Adiciona padding AO FIM para que inputSeq e targetSeq tenham o mesmo comprimento
                    // até currentBatchTargetLen (que é o maxSequenceLength global)
                    while (inputSeq.Count < currentBatchTargetLen -1) { inputSeq.Add(padTokenId); } // Pad Input
                    while (targetSeq.Count < currentBatchTargetLen -1) { targetSeq.Add(padTokenId); } // Pad Target

                    // Garante que o input e o target tenham exatamente o mesmo tamanho após padding/truncamento
                    // e que não excedam maxSequenceLength-1 antes do último token/padding.
                     // Input será tokens [0] a [max-2], target será [1] a [max-1]
                     inputSeq = inputSeq.Take(maxSequenceLength -1).ToList();
                     targetSeq = targetSeq.Take(maxSequenceLength -1).ToList();

                    // Final length should be maxSequenceLength - 1 for input/target
                    // And the model's internal forward pass will effectively predict the token at index [i+1] based on [0..i]
                    // The output logits will be for sequence positions [0..maxSequenceLength-2] predicting [1..maxSequenceLength-1]

                    // Adiciona o par ao batch
                    inputBatchList.Add(inputSeq.Select(id => (long)id).ToArray());
                    targetBatchList.Add(targetSeq.Select(id => (long)id).ToArray());
                }

                // Se o batch ficou vazio após filtrar sequências curtas, pula
                if (!inputBatchList.Any()) continue;


                // *** Converte as listas de arrays em tensores TorchSharp ***
                // Stack dimension: 0 (cria a dimensão de batch)
                try
                {
                    // Stacks uma lista de long[] em um tensor (Batch, SeqLen)
                    var inputTensor = torch.stack(inputBatchList.Select(arr => tensor(arr, dtype: ScalarType.Int64)).ToList());
                    var targetTensor = torch.stack(targetBatchList.Select(arr => tensor(arr, dtype: ScalarType.Int64)).ToList());

                    // Verifica shapes resultantes
                    // Console.WriteLine($"Created batch {batches.Count + 1}: Input shape {inputTensor.shape}, Target shape {targetTensor.shape}");

                    batches.Add(Tuple.Create(inputTensor, targetTensor));
                }
                catch (Exception tensorEx)
                {
                     Report($"Error creating tensor batch {batches.Count + 1}: {tensorEx.Message}. Skipping batch.");
                     // Garante dispose de quaisquer tensores criados parcialmente neste batch
                     inputBatchList.Select(arr => tensor(arr, dtype: ScalarType.Int64)).ToList().ForEach(t => t.Dispose());
                     targetBatchList.Select(arr => tensor(arr, dtype: ScalarType.Int64)).ToList().ForEach(t => t.Dispose());
                }
            }

            Report($"Finished batch creation. Final number of batches: {batches.Count}.");
            return batches;
        }
    } // --- Fim da Classe TransformerTrainer ---
} // --- Fim do Namespace ---