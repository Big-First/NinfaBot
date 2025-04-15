namespace ChatBotAPI.Core;

// Rede neural simples
public class NeuralNetwork
{
    private readonly DenseLayer[] layers;

    public NeuralNetwork(params int[] layerSizes)
    {
        if (layerSizes.Length < 2)
            throw new ArgumentException("A rede deve ter pelo menos uma camada de entrada e uma de saída.");

        layers = new DenseLayer[layerSizes.Length - 1];
        for (int i = 0; i < layers.Length; i++)
        {
            bool useSigmoid = (i == layers.Length - 1); // Sigmoide apenas na última camada
            layers[i] = new DenseLayer(layerSizes[i], layerSizes[i + 1], useSigmoid);
        }
    }

    public Tensor Forward(Tensor input)
    {
        Tensor current = input;
        foreach (var layer in layers)
            current = layer.Forward(current);
        return current;
    }
}