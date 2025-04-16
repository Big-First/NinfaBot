namespace ChatBotAPI.Core
{
    public class BinaryTreeNeuralModel : Model
    {
        private Node root;
        private readonly int embeddingSize;
        private readonly int vocabSize;
        private double[,] embeddings;
        private readonly int maxSequenceLength;
        private readonly Random rand;

        public BinaryTreeNeuralModel(int vocabSize, int embeddingSize, int maxSequenceLength)
        {
            this.vocabSize = vocabSize;
            this.embeddingSize = embeddingSize;
            this.maxSequenceLength = maxSequenceLength;
            root = new Node(embeddingSize);
            rand = new Random();

            embeddings = new double[vocabSize, embeddingSize];
            for (int i = 0; i < vocabSize; i++)
            {
                for (int j = 0; j < embeddingSize; j++)
                {
                    embeddings[i, j] = rand.NextDouble() * 0.2 - 0.1;
                }
            }
        }

        public override void Initialize(int maxDepth)
        {
            InitializeTree(root, 0, maxDepth);
        }

        private void InitializeTree(Node node, int depth, int maxDepth)
        {
            if (depth >= maxDepth) return;
            node.Left = new Node(embeddingSize);
            node.Right = new Node(embeddingSize);
            InitializeTree(node.Left, depth + 1, maxDepth);
            InitializeTree(node.Right, depth + 1, maxDepth);
        }

        public override double[] Predict(int[] input)
        {
            double[][] sequenceEmbeddings = new double[input.Length][];
            for (int i = 0; i < input.Length; i++)
            {
                sequenceEmbeddings[i] = new double[embeddingSize];
                for (int j = 0; j < embeddingSize; j++)
                {
                    sequenceEmbeddings[i][j] = embeddings[input[i], j];
                }
            }

            double[] contextVector = Attention(sequenceEmbeddings);

            double[] output = new double[vocabSize];
            Node current = root;
            for (int i = 0; i < vocabSize; i++)
            {
                double sum = current.Bias;
                for (int j = 0; j < embeddingSize; j++)
                {
                    sum += contextVector[j] * current.Weights[j];
                }
                output[i] = ReLU(sum);
                current = rand.Next(2) == 0 ? current.Left : current.Right;
                if (current == null) break;
            }

            return Softmax(output);
        }

        public override void Train(int[] input, int[] target)
        {
            double[][] sequenceEmbeddings = new double[input.Length][];
            for (int i = 0; i < input.Length; i++)
            {
                sequenceEmbeddings[i] = new double[embeddingSize];
                for (int j = 0; j < embeddingSize; j++)
                {
                    sequenceEmbeddings[i][j] = embeddings[input[i], j];
                }
            }

            double[] contextVector = Attention(sequenceEmbeddings);

            double[] output = new double[vocabSize];
            Node current = root;
            for (int i = 0; i < vocabSize; i++)
            {
                double sum = current.Bias;
                for (int j = 0; j < embeddingSize; j++)
                {
                    sum += contextVector[j] * current.Weights[j];
                }
                output[i] = ReLU(sum);
                current = rand.Next(2) == 0 ? current.Left : current.Right;
                if (current == null) break;
            }
            output = Softmax(output);

            double learningRate = 0.01;
            for (int i = 0; i < vocabSize; i++)
            {
                double error = target[i] - output[i];
                current = root;
                for (int j = 0; j < embeddingSize; j++)
                {
                    current.Weights[j] += learningRate * error * contextVector[j];
                }
                current.Bias += learningRate * error;

                for (int k = 0; k < input.Length; k++)
                {
                    for (int j = 0; j < embeddingSize; j++)
                    {
                        embeddings[input[k], j] += learningRate * error * sequenceEmbeddings[k][j];
                    }
                }
            }
        }

        private double ReLU(double x) => Math.Max(0, x);

        private double[] Softmax(double[] input)
        {
            double max = input.Max();
            double sum = 0;
            double[] result = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                result[i] = Math.Exp(input[i] - max);
                sum += result[i];
            }
            for (int i = 0; i < input.Length; i++)
            {
                result[i] /= sum;
            }
            return result;
        }

        private double[] Attention(double[][] sequenceEmbeddings)
        {
            int seqLen = sequenceEmbeddings.Length;
            double[] attentionScores = new double[seqLen];
            double[] output = new double[embeddingSize];

            for (int i = 0; i < seqLen; i++)
            {
                attentionScores[i] = 0;
                for (int j = 0; j < embeddingSize; j++)
                {
                    attentionScores[i] += sequenceEmbeddings[i][j] * sequenceEmbeddings[i][j];
                }
                attentionScores[i] /= Math.Sqrt(embeddingSize);
            }

            attentionScores = Softmax(attentionScores);

            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < embeddingSize; j++)
                {
                    output[j] += attentionScores[i] * sequenceEmbeddings[i][j];
                }
            }

            return output;
        }
    }
}