namespace ChatBotAPI.Core;

public class Node
{
    public double Value { get; set; }
    public Node Left { get; set; }
    public Node Right { get; set; }
    public double[] Weights { get; set; }
    public double Bias { get; set; }

    public Node(int weightSize)
    {
        Weights = new double[weightSize];
        Random rand = new Random();
        for (int i = 0; i < weightSize; i++)
        {
            Weights[i] = rand.NextDouble() * 0.1;
        }
        Bias = rand.NextDouble() * 0.1;
    }
}