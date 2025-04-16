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

        public Trainer(NeuralModel model, Tokenizer tokenizer /*, int maxDepth*/)
        {
            this.model = model ?? throw new ArgumentNullException(nameof(model));
            this.tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
            // this.maxDepth = maxDepth;
        }

        public void Train(List<string> trainingData, int epochs)
        {
            // Removido model.Initialize(maxDepth); se não existir mais

            // Expõe PadTokenId no Tokenizer para poder usá-lo aqui
            int padTokenId = tokenizer.PadTokenId; // Supondo que Tokenizer tenha essa propriedade pública

            Console.WriteLine($"Starting training with PadTokenId = {padTokenId}...");

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                Console.WriteLine($"Epoch {epoch + 1}/{epochs}");
                int steps = 0;
                foreach (string sentence in trainingData)
                {
                    if (string.IsNullOrWhiteSpace(sentence)) continue;

                    int[] tokenIds = tokenizer.Tokenize(sentence);

                    // Para cada posição na sequência (exceto a última),
                    // use os tokens até essa posição como entrada
                    // e o token seguinte como alvo.
                    for (int i = 0; i < tokenIds.Length - 1; i++)
                    {
                        // Ignora se o token atual ou o próximo for PAD
                        if (tokenIds[i] == padTokenId || tokenIds[i+1] == padTokenId)
                        {
                            // Podemos parar aqui para esta sentença, pois chegamos ao padding
                             // ou se quisermos treinar com PAD como entrada/saída, remova esta condição.
                             // Por simplicidade, vamos parar no primeiro PAD encontrado.
                             break;
                        }

                        // A entrada são todos os tokens até a posição 'i' (inclusive)
                        int[] currentInput = tokenIds.Take(i + 1).ToArray();

                        // O alvo é o token na posição 'i + 1'
                        int targetIndex = tokenIds[i + 1];

                        // Chama o método Train do modelo com a assinatura correta
                        model.Train(currentInput, targetIndex);
                        steps++;

                    } // fim do loop interno (posições na sentença)
                } // fim do loop externo (sentenças)
                Console.WriteLine($"Epoch {epoch + 1} completed with {steps} training steps.");
                if (steps == 0) {
                    Console.WriteLine("Warning: No valid training steps were performed in this epoch. Check training data and padding.");
                }

            } // fim do loop de épocas
            Console.WriteLine("Training finished.");
        }
    }
}