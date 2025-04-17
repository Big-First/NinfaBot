// Trainer.cs - VERSÃO CORRETA E COMPLETA

using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics; // Para Stopwatch
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

        // --- CONSTRUTOR ---
        public Trainer(TorchSharpModel model, Tokenizer tokenizer, double learningRate = 0.001)
        {
            this.model = model ?? throw new ArgumentNullException(nameof(model));
            this.tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
            this.device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            Console.WriteLine($"Trainer using device: {this.device.type}");
            this.model.to(this.device);
            this.optimizer = Adam(this.model.parameters(), lr: learningRate);
            int padTokenId = this.tokenizer.PadTokenId;
            this.lossFunction = CrossEntropyLoss(ignore_index: padTokenId).to(this.device);
            Console.WriteLine($"Trainer initialized loss function. Device: {this.device.type}. ignore_index (PadTokenId): {padTokenId}");
             if (tokenizer.ActualVocabSize <= 2) {
                 Console.Error.WriteLine("CRITICAL WARNING: Trainer detected Tokenizer ActualVocabSize <= 2. Vocabulary likely failed to load or is invalid.");
             }
        }

        // --- MÉTODO Train ---
        public void Train(List<string> trainingData, int epochs)
        {
            // Verificações iniciais
            if (trainingData == null || !trainingData.Any()) { Console.WriteLine("Training data is empty. Skipping training."); return; }
            if (tokenizer.ActualVocabSize <= 2) { Console.Error.WriteLine("Cannot train: Tokenizer vocabulary is too small (Size <= 2). Check vocabulary loading."); return; }

            int padTokenId = tokenizer.PadTokenId;
            int vocabSize = tokenizer.ActualVocabSize;
            int maxSeqLen = 50; // Valor padrão, ajuste se necessário ou use GetMaxSequenceLength()
             try { maxSeqLen = tokenizer.GetMaxSequenceLength(); }
             catch { Console.WriteLine($"Warning: Could not get MaxSequenceLength from tokenizer. Using fallback={maxSeqLen}."); }

            // Log inicial (APENAS UMA VEZ)
            Console.WriteLine($"Starting TorchSharp training on {device.type}. Epochs: {epochs}. Sentences: {trainingData.Count}. PadTokenId: {padTokenId}. VocabSize: {vocabSize}. MaxSeqLen: {maxSeqLen}.");
            Stopwatch epochStopwatch = new Stopwatch();

            // --- Loop Principal de Épocas ---
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                model.train(); // MODO TREINO
                epochStopwatch.Restart();
                Console.WriteLine($"--- Epoch {epoch + 1}/{epochs} ---");
                float totalLoss = 0f; // Usa float para consistência
                long totalStepsInEpoch = 0;
                long skippedSteps = 0;
                long sentenceCount = 0;

                // --- Loop pelas Sentenças ---
                try // Catch em volta do loop de sentenças
                {
                    foreach (string sentence in trainingData)
                    {
                        // Console.WriteLine($"DEBUG: Epoch {epoch + 1}, Sentence Loop Start. Processing sentence {sentenceCount + 1}..."); // Log Opcional
                        sentenceCount++;
                        int[] tokens = tokenizer.Tokenize(sentence);
                        // Console.WriteLine($"DEBUG: Epoch {epoch + 1}, Sentence {sentenceCount}: Tokenization finished. Length={tokens.Length}."); // Log Opcional

                        if (tokens.Length <= 1) {
                            Console.WriteLine($"AVISO: Epoch {epoch + 1}, Sent {sentenceCount} ('{sentence.Substring(0, Math.Min(sentence.Length, 30))}...') resulted in tokens.Length <= 1. Skipping.");
                            skippedSteps += tokens.Length;
                            continue;
                        }
                        // Console.WriteLine($"DEBUG: Epoch {epoch + 1}, Sentence {sentenceCount}: Entering inner step loop (i=1 to {tokens.Length - 1})..."); // Log Opcional

                        // --- Loop Interno pelos Passos ---
                        for (int i = 1; i < tokens.Length; i++)
                        {
                            if (tokens[i] == padTokenId) {
                                skippedSteps++;
                                continue;
                            }

                            long[] inputSequence = tokens.Take(i).Select(t => (long)t).ToArray();
                            long targetTokenId = tokens[i];

                            if (!inputSequence.Any()) {
                                skippedSteps++;
                                continue;
                            }

                            Tensor? inputTensor = null;
                            Tensor? targetTensor = null;
                            Tensor? outputLogits = null;
                            Tensor? loss = null;

                            try // <- TRY INTERNO DO PASSO DE TREINAMENTO
                            {
                                inputTensor = tensor(inputSequence, dtype: ScalarType.Int64).to(device);
                                targetTensor = tensor(new long[] { targetTokenId }, dtype: ScalarType.Int64).to(device);

                                optimizer.zero_grad();

                                // Console.WriteLine($"DEBUG: Step i={i}: BEFORE model.forward"); // Log Opcional
                                outputLogits = model.forward(inputTensor);
                                // Console.WriteLine($"DEBUG: Step i={i}: AFTER model.forward. outputLogits is null = {outputLogits == null}"); // Log Opcional

                                if ((bool)(outputLogits == null)) {
                                    Console.WriteLine($"WARN: Step i={i}: outputLogits IS NULL. Skipping step.");
                                    skippedSteps++;
                                    inputTensor?.Dispose(); targetTensor?.Dispose();
                                    continue;
                                }

                                // Verificação de Shape (sem try-catch extra)
                                // Console.WriteLine($"DEBUG: Step i={i}: Checking shape. Shape is [{string.Join(",", outputLogits.shape)}], Expecting [{vocabSize}]"); // Log Opcional
                                bool isShapeOk = (outputLogits.dim() == 1 && outputLogits.shape[0] == vocabSize);
                                // Console.WriteLine($"DEBUG: Step i={i}: Shape check result. isShapeOk = {isShapeOk}"); // Log Opcional

                                if (!isShapeOk) {
                                    Console.WriteLine($"WARN: Step i={i}: Shape IS NOT OK (Got: [{string.Join(",", outputLogits.shape)}], Expected: [{vocabSize}]). Skipping step.");
                                    skippedSteps++;
                                    inputTensor?.Dispose(); targetTensor?.Dispose(); outputLogits?.Dispose();
                                    continue;
                                }
                                // Console.WriteLine($"DEBUG: Step i={i}: Shape OK. Calculating loss..."); // Log Opcional

                                // Calcular Loss
                                loss = lossFunction.forward(outputLogits!.unsqueeze(0), targetTensor!);
                                float currentLoss = loss!.item<float>();
                                // Console.WriteLine($"DEBUG: Step i={i}: Loss = {currentLoss:F6}. Performing backward/step..."); // Log Opcional

                                // Backward e Step
                                loss.backward();
                                optimizer.step();

                                // Acumular e Contar
                                totalLoss += currentLoss;
                                totalStepsInEpoch++; // <-- Contador de passos válidos
                                // Console.WriteLine($"DEBUG: Step i={i}: Step successful!"); // Log Opcional

                            }
                            catch (Exception stepEx) // <- CATCH INTERNO DO PASSO DE TREINAMENTO
                            {
                                Console.Error.WriteLine($"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
                                Console.Error.WriteLine($"ERRO FATAL (Catch Block) no passo i={i}, Sent {sentenceCount}, Epoch {epoch + 1}: {stepEx.Message}");
                                Console.Error.WriteLine($"Detalhes da Exceção: {stepEx.ToString()}");
                                // ... (logs de contexto adicionais) ...
                                Console.Error.WriteLine($"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
                                skippedSteps++;
                            }
                            finally
                            {
                               inputTensor?.Dispose(); targetTensor?.Dispose(); outputLogits?.Dispose(); loss?.Dispose();
                            }
                        } // Fim loop interno (passos)
                        // Console.WriteLine($"DEBUG: Epoch {epoch + 1}, Sentence {sentenceCount}: Finished inner step loop."); // Log Opcional

                    } // Fim loop externo (sentenças)
                }
                catch (Exception exOuterLoop) // Pega erros no loop de sentenças (raro)
                {
                    Console.Error.WriteLine($"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
                    Console.Error.WriteLine($"ERRO FATAL NO LOOP EXTERNO (Epoch {epoch + 1}, próximo à sentença {sentenceCount + 1}): {exOuterLoop.Message}");
                    Console.Error.WriteLine($"Detalhes da Exceção: {exOuterLoop.ToString()}");
                    Console.Error.WriteLine($"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
                }

                // --- Fim da Época ---
                // Código para calcular e imprimir o resumo da época (NO LUGAR CORRETO)
                epochStopwatch.Stop();
                float avgLoss = totalStepsInEpoch > 0 ? totalLoss / totalStepsInEpoch : 0f;
                Console.WriteLine($"--- Epoch {epoch + 1} completed in {epochStopwatch.ElapsedMilliseconds} ms. Avg Loss: {avgLoss:F6}, Steps: {totalStepsInEpoch}, Skipped: {skippedSteps} ---");

            } // --- Fim do Loop Principal de Épocas ---

            model.eval(); // MODO AVALIAÇÃO
            Console.WriteLine("Training finished.");
        } // --- Fim do Método Train ---
    } // --- Fim da Classe Trainer ---
} // --- Fim do Namespace ---