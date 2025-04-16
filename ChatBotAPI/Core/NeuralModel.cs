namespace ChatBotAPI.Core
{
    public interface NeuralModel
    {
        double[] Predict(int[] input);
        void Train(int[] input, int targetTokenIndex);
    }
}