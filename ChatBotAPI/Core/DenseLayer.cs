namespace ChatBotAPI.Core;

// Camada densa (fully connected)
public class DenseLayer
{
    public Tensor Weights { get; private set; }
    public Tensor Bias { get; private set; }
    private readonly bool useSigmoid;

    public DenseLayer(int inputSize, int outputSize, bool useSigmoid = false)
    {
        this.useSigmoid = useSigmoid;
        // Inicialização dos pesos com valores aleatórios (distribuição normal simplificada)
        var random = new Random();
        float[] weightData = new float[inputSize * outputSize];
        for (int i = 0; i < weightData.Length; i++)
            weightData[i] = (float)(random.NextDouble() * 0.2 - 0.1); // Pequenos valores aleatórios
        Weights = Tensor.FromData(weightData, inputSize, outputSize);

        // Inicializar bias com zeros
        Bias = Tensor.Zeros(outputSize);
    }

    public Tensor Forward(Tensor input)
    {
        // Verificar compatibilidade
        if (input.Shape.Last() != Weights.Shape[0])
            throw new ArgumentException("A última dimensão do input deve corresponder à primeira dimensão dos pesos.");

        // Multiplicação matriz-vetor: output = input @ weights + bias
        int batchSize = input.Rank == 1 ? 1 : input.Shape[0];
        int outputSize = Weights.Shape[1];
        Tensor result;

        if (input.Rank == 1)
        {
            result = new Tensor(outputSize);
            for (int j = 0; j < outputSize; j++)
            {
                float sum = 0;
                for (int k = 0; k < input.Size; k++)
                    sum += input[k] * Weights[k, j];
                result[j] = sum + Bias[j];
            }
        }
        else
        {
            result = new Tensor(batchSize, outputSize);
            for (int i = 0; i < batchSize; i++)
            for (int j = 0; j < outputSize; j++)
            {
                float sum = 0;
                for (int k = 0; k < input.Shape[1]; k++)
                    sum += input[i, k] * Weights[k, j];
                result[i, j] = sum + Bias[j];
            }
        }

        // Aplicar ativação
        return useSigmoid ? Activations.Sigmoid(result) : Activations.ReLU(result);
    }
}