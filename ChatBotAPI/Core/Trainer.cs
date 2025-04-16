using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;

namespace ChatBotAPI.Core
{
    public class Trainer
    {
        private readonly NeuralModel model;
        private readonly Tokenizer tokenizer; // Tokenizer está disponível aqui
        private const bool LogTrainerSteps = false;

        public Trainer(NeuralModel model, Tokenizer tokenizer)
        {
            this.model = model ?? throw new ArgumentNullException(nameof(model));
            this.tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
        }

        public void Train(List<string> trainingData, int epochs)
        {
             // ... (código inicial, obter padTokenId) ...
             int padTokenId = tokenizer.PadTokenId;
             Console.WriteLine($"Starting training with PadTokenId = {padTokenId}...");

            Stopwatch epochStopwatch = new Stopwatch();

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                epochStopwatch.Restart();
                Console.WriteLine($"--- Epoch {epoch + 1}/{epochs} ---");
                long totalStepsInEpoch = 0;
                long skippedSteps = 0;

                foreach (string sentence in trainingData)
                {
                    // ... (tokenizar sentença) ...
                    int[] tokenIds = tokenizer.Tokenize(sentence);

                    for (int i = 0; i < tokenIds.Length - 1; i++)
                    {
                        // ... (verificar padding) ...
                        if (tokenIds[i] == padTokenId || tokenIds[i+1] == padTokenId) {
                            skippedSteps += (tokenIds.Length - 1 - i);
                            break;
                        }

                        int[] currentInput = tokenIds.Take(i + 1).ToArray();
                        int targetIndex = tokenIds[i + 1];

                        // *** CORREÇÃO: Usa tokenizer.ActualVocabSize para verificar o limite ***
                        if (targetIndex < 0 || targetIndex >= tokenizer.ActualVocabSize) {
                             Console.WriteLine($"Warning: Invalid target index {targetIndex} (Vocab Size: {tokenizer.ActualVocabSize}) generated for input sequence (length {currentInput.Length}). Skipping step.");
                             skippedSteps++;
                             continue; // Pula esta etapa de treino específica
                        }

                        // ... (log opcional) ...
                        if(LogTrainerSteps) {
                            // Usa o Detokenize do tokenizer disponível
                             Console.WriteLine($"  Epoch {epoch+1}, Step {totalStepsInEpoch+1}: InputLen={currentInput.Length}, Target={targetIndex} ({this.tokenizer.Detokenize(new int[]{targetIndex})})");
                        }

                        model.Train(currentInput, targetIndex);
                        totalStepsInEpoch++;

                    } // Fim loop interno
                } // Fim loop externo

                epochStopwatch.Stop();
                Console.WriteLine($"--- Epoch {epoch + 1} completed in {epochStopwatch.ElapsedMilliseconds} ms. Total steps: {totalStepsInEpoch}, Skipped/Padding steps: {skippedSteps} ---");

                 if (totalStepsInEpoch == 0 && trainingData.Any(s => !string.IsNullOrWhiteSpace(s))) {
                    Console.WriteLine("Warning: No valid training steps were performed in this epoch.");
                }
            }
            Console.WriteLine("Training finished.");
        }
    }
}