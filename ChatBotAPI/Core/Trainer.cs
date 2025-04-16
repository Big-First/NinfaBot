using System;
using System.Collections.Generic;

namespace ChatBotAPI.Core
{
    public class Trainer
    {
        private readonly NeuralModel model;
        private readonly Tokenizer tokenizer;
        private readonly int maxDepth;

        public Trainer(NeuralModel model, Tokenizer tokenizer, int maxDepth)
        {
            this.model = model;
            this.tokenizer = tokenizer;
            this.maxDepth = maxDepth;
        }

        public void Train(List<string> trainingData, int epochs)
        {
            model.Initialize(maxDepth);

            Random rand = new Random();
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                Console.WriteLine($"Epoch {epoch + 1}/{epochs}");
                foreach (string data in trainingData)
                {
                    int[] inputTokens = tokenizer.Tokenize(data);
                    int[] targetTokens = new int[inputTokens.Length];
                    Array.Copy(inputTokens, 0, targetTokens, 1, inputTokens.Length - 1);
                    targetTokens[0] = tokenizer.Tokenize("<PAD>")[0];

                    int[] target = new int[inputTokens.Length];
                    for (int i = 0; i < target.Length; i++)
                    {
                        target[i] = targetTokens[i];
                    }

                    model.Train(inputTokens, target);
                }
            }
        }
    }
}