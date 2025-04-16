using System;
using System.Collections.Generic;
using System.Linq;

namespace ChatBotAPI.Core
{
    // Removido : NeuralModel -> Vamos implementar a interface depois se necessário,
    // focando na lógica correta primeiro. Se precisar herdar, adicione : NeuralModel de volta.
    public class BinaryTreeNeuralModel : NeuralModel // Renomear talvez para SimpleAttentionModel ?
    {
        // Removida a estrutura de árvore: private Node root;
        private readonly int embeddingSize;
        private readonly int vocabSize; // Tamanho real do vocabulário usado
        private double[,] embeddings; // Shape: [vocabSize, embeddingSize]
        private readonly int maxSequenceLength;
        private readonly Random rand;

        // --- Novos Parâmetros para a Camada Linear de Saída ---
        private double[,] outputWeights; // Shape: [embeddingSize, vocabSize]
        private double[] outputBias;    // Shape: [vocabSize]
        // --- Fim dos Novos Parâmetros ---

        // Construtor Atualizado: Não precisa mais do Model config, recebe vocabSize diretamente
        public BinaryTreeNeuralModel(int actualVocabSize, int embeddingSize, int maxSequenceLength)
        {
            if (actualVocabSize <= 0) throw new ArgumentException("Vocabulary size must be positive.", nameof(actualVocabSize));
            if (embeddingSize <= 0) throw new ArgumentException("Embedding size must be positive.", nameof(embeddingSize));

            this.vocabSize = actualVocabSize; // Usa o tamanho real passado
            this.embeddingSize = embeddingSize;
            this.maxSequenceLength = maxSequenceLength; // Ainda relevante para padding/truncamento? A atenção lida com comprimentos variáveis.
            this.rand = new Random();

            // Inicializa Embeddings
            this.embeddings = new double[this.vocabSize, this.embeddingSize];
            // Inicializa Pesos da Camada de Saída
            this.outputWeights = new double[this.embeddingSize, this.vocabSize];
            // Inicializa Bias da Camada de Saída
            this.outputBias = new double[this.vocabSize];

            // Preenche com valores aleatórios pequenos (exemplo: Xavier/Glorot initialization simplificada)
            double limitEmbed = Math.Sqrt(6.0 / (this.vocabSize + this.embeddingSize));
            for (int i = 0; i < this.vocabSize; i++)
            {
                for (int j = 0; j < this.embeddingSize; j++)
                {
                    this.embeddings[i, j] = (rand.NextDouble() * 2 - 1) * limitEmbed;
                }
            }

            double limitOutput = Math.Sqrt(6.0 / (this.embeddingSize + this.vocabSize));
             for (int i = 0; i < this.embeddingSize; i++)
             {
                 for (int j = 0; j < this.vocabSize; j++)
                 {
                    this.outputWeights[i, j] = (rand.NextDouble() * 2 - 1) * limitOutput;
                 }
             }
            // Bias pode ser inicializado com zero ou pequenos valores aleatórios
            for(int i = 0; i < this.vocabSize; i++)
            {
                this.outputBias[i] = 0; // Inicializar bias com zero é comum
            }

             Console.WriteLine($"Model Initialized: VocabSize={this.vocabSize}, EmbeddingSize={this.embeddingSize}");
             Console.WriteLine($"Shapes: Embeddings[{this.embeddings.GetLength(0)},{this.embeddings.GetLength(1)}], OutputWeights[{this.outputWeights.GetLength(0)},{this.outputWeights.GetLength(1)}], OutputBias[{this.outputBias.Length}]");
        }

        // Removido Initialize e InitializeTree

        // Método Predict Refatorado
        // public override double[] Predict(int[] input) // Se herdando
        public double[] Predict(int[] input)
        {
            if (input == null || input.Length == 0)
            {
                 // Retorna uma distribuição uniforme ou lança exceção?
                 // Por agora, retornamos uma distribuição uniforme como placeholder.
                Console.WriteLine("Warning: Predict called with empty input.");
                var uniformDist = new double[this.vocabSize];
                Array.Fill(uniformDist, 1.0 / this.vocabSize);
                return uniformDist;
            }

            // 1. Obter Embeddings da Sequência de Entrada
            // Filtra tokens inválidos (ex: padding se não for tratado antes, ou índices fora do limite)
            var validInputs = input.Where(idx => idx >= 0 && idx < this.vocabSize).ToArray();
             if (validInputs.Length == 0) {
                Console.WriteLine("Warning: Predict called with input containing only invalid token indices.");
                var uniformDist = new double[this.vocabSize];
                Array.Fill(uniformDist, 1.0 / this.vocabSize);
                return uniformDist;
            }

            double[][] sequenceEmbeddings = new double[validInputs.Length][];
            for (int i = 0; i < validInputs.Length; i++)
            {
                sequenceEmbeddings[i] = new double[embeddingSize];
                int tokenIndex = validInputs[i];
                for (int j = 0; j < embeddingSize; j++)
                {
                    // Copia a linha correspondente da matriz de embeddings
                    sequenceEmbeddings[i][j] = embeddings[tokenIndex, j];
                }
            }

            // 2. Calcular Vetor de Contexto usando Atenção
            double[] contextVector = Attention(sequenceEmbeddings);

            // 3. Calcular Logits usando a Camada Linear
            double[] logits = new double[this.vocabSize];
            for (int i = 0; i < this.vocabSize; i++) // Para cada palavra no vocabulário de saída
            {
                double sum = 0;
                for (int j = 0; j < this.embeddingSize; j++) // Dot product com a coluna i dos pesos
                {
                    sum += contextVector[j] * outputWeights[j, i];
                }
                logits[i] = sum + outputBias[i]; // Adiciona o bias
            }

            // 4. Aplicar Softmax para obter Probabilidades
            // O ReLU foi removido daqui, Softmax opera sobre logits
            double[] outputProbabilities = Softmax(logits);

            return outputProbabilities;
        }

        // Método Train SIGNIFICATIVAMENTE SIMPLIFICADO
        // Assume que 'targetTokenIndex' é o índice da *única* palavra correta esperada como saída.
        // ATENÇÃO: Este método NÃO treina os embeddings nem a camada de atenção.
        //          Apenas atualiza a camada linear final (outputWeights e outputBias).
        // public override void Train(int[] input, int[] target) // Se herdando
        public void Train(int[] input, int targetTokenIndex) // Modificado para receber índice único
        {
             if (input == null || input.Length == 0) {
                Console.WriteLine("Warning: Train called with empty input. Skipping.");
                return;
            }
             if (targetTokenIndex < 0 || targetTokenIndex >= this.vocabSize) {
                 Console.WriteLine($"Warning: Train called with invalid target index {targetTokenIndex}. Skipping.");
                 return;
             }

            // --- Forward Pass (igual ao Predict) ---
             var validInputs = input.Where(idx => idx >= 0 && idx < this.vocabSize).ToArray();
             if (validInputs.Length == 0) {
                Console.WriteLine("Warning: Train called with input containing only invalid token indices. Skipping.");
                return;
            }
            double[][] sequenceEmbeddings = new double[validInputs.Length][];
            for (int i = 0; i < validInputs.Length; i++) {
                sequenceEmbeddings[i] = new double[embeddingSize];
                int tokenIndex = validInputs[i];
                for (int j = 0; j < embeddingSize; j++) {
                    sequenceEmbeddings[i][j] = embeddings[tokenIndex, j];
                }
            }
            double[] contextVector = Attention(sequenceEmbeddings);
            double[] logits = new double[this.vocabSize];
            for (int i = 0; i < this.vocabSize; i++) {
                double sum = 0;
                for (int j = 0; j < this.embeddingSize; j++) {
                    sum += contextVector[j] * outputWeights[j, i];
                }
                logits[i] = sum + outputBias[i];
            }
            double[] outputProbabilities = Softmax(logits);
            // --- Fim do Forward Pass ---


            // --- Backpropagation Simplificada (Apenas para Camada de Saída) ---
            double learningRate = 0.01; // Taxa de aprendizado (pode ser configurável)

            // Calcula o gradiente dos logits (assumindo Cross-Entropy Loss)
            // dL/dLogits = outputProbabilities - target (onde target é one-hot)
            // Se targetTokenIndex é o índice correto, o vetor target one-hot tem 1 nessa posição e 0 nas outras.
            double[] dLogits = new double[this.vocabSize];
            for(int i = 0; i < this.vocabSize; i++)
            {
                dLogits[i] = outputProbabilities[i];
            }
            dLogits[targetTokenIndex] -= 1.0; // Subtrai 1 da posição correta

            // Atualiza Pesos e Bias da Camada de Saída
            for (int i = 0; i < this.vocabSize; i++) // Para cada neurônio de saída
            {
                 // Gradiente do Bias: dL/dBias[i] = dLogits[i]
                 outputBias[i] -= learningRate * dLogits[i];

                 // Gradiente dos Pesos: dL/dWeight[j,i] = contextVector[j] * dLogits[i]
                 for (int j = 0; j < this.embeddingSize; j++) // Para cada entrada no contextVector
                 {
                    outputWeights[j, i] -= learningRate * contextVector[j] * dLogits[i];
                 }
            }

            // !! ATENÇÃO !!
            // A retropropagação para a camada de Atenção e para os Embeddings NÃO está implementada aqui.
            // Para um treinamento completo, seria necessário calcular dL/dContextVector e propagar
            // esse gradiente para trás através da atenção e para os embeddings.
            // Isso é significativamente mais complexo.
        }

        // Removido ReLU (não usado antes do Softmax)
        // private double ReLU(double x) => Math.Max(0, x);

        private double[] Softmax(double[] input)
        {
            if (input == null || input.Length == 0) return Array.Empty<double>();

            double maxLogit = input[0];
            for(int i = 1; i < input.Length; i++) {
                if (input[i] > maxLogit) maxLogit = input[i];
            }
            // double maxLogit = input.Max(); // Linq pode ser mais lento para arrays grandes

            double sumExp = 0;
            double[] result = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                // Subtrair maxLogit para estabilidade numérica
                result[i] = Math.Exp(input[i] - maxLogit);
                sumExp += result[i];
            }

            if (sumExp == 0 || double.IsNaN(sumExp) || double.IsInfinity(sumExp)) {
                 // Evitar divisão por zero ou NaN. Retorna distribuição uniforme em caso de problema.
                 Console.Error.WriteLine("Warning: Softmax sum is zero, NaN or Infinity. Returning uniform distribution.");
                 Array.Fill(result, 1.0 / input.Length);
                 return result;
            }

            for (int i = 0; i < input.Length; i++)
            {
                result[i] /= sumExp;
            }
            return result;
        }

        // Atenção permanece a mesma por enquanto (simplificada)
        private double[] Attention(double[][] sequenceEmbeddings)
        {
            if (sequenceEmbeddings == null || sequenceEmbeddings.Length == 0)
            {
                return new double[embeddingSize]; // Retorna vetor de zeros se não houver embeddings
            }

            int seqLen = sequenceEmbeddings.Length;
            double[] attentionScores = new double[seqLen];
            double[] outputContext = new double[embeddingSize]; // Inicializado com zeros

            // Calcula scores brutos (simplificado: norma L2^2 / sqrt(d_k))
            double scale = Math.Sqrt(embeddingSize);
            if (scale == 0) scale = 1.0; // Evita divisão por zero

            for (int i = 0; i < seqLen; i++)
            {
                double score = 0;
                for (int j = 0; j < embeddingSize; j++)
                {
                    score += sequenceEmbeddings[i][j] * sequenceEmbeddings[i][j];
                }
                attentionScores[i] = score / scale;
            }

            // Normaliza scores com Softmax
            attentionScores = Softmax(attentionScores);

            // Calcula o vetor de contexto ponderado
            for (int i = 0; i < seqLen; i++) // Para cada posição na sequência
            {
                for (int j = 0; j < embeddingSize; j++) // Para cada dimensão do embedding
                {
                    outputContext[j] += attentionScores[i] * sequenceEmbeddings[i][j];
                }
            }

            return outputContext;
        }

         // Função auxiliar para Dot Product (pode ser útil se precisar em outros lugares)
         private double DotProduct(double[] vec1, double[] vec2) {
            if(vec1.Length != vec2.Length) throw new ArgumentException("Vectors must have the same length.");
            double sum = 0;
            for(int i = 0; i < vec1.Length; i++) {
                sum += vec1[i] * vec2[i];
            }
            return sum;
         }
    }
}