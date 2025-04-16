namespace ChatBotAPI.Core
{
    public abstract class Model
    {
        public abstract void Initialize(int maxDepth);
        public abstract double[] Predict(int[] input);
        public abstract void Train(int[] input, int[] target);
    }
}