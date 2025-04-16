namespace ChatBotAPI.Core
{
    public abstract class NeuralModel
    {
        public abstract void Initialize(int maxDepth);
        public abstract double[] Predict(int[] input);
        public abstract void Train(int[] input, int[] target);
    }
}