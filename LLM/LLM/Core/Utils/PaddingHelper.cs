using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;

namespace LLM.Core.Utils; 

public static class PaddingHelper
{
    /// <summary>
    /// Adiciona padding a uma lista de tokens para atingir um comprimento específico.
    /// Trunca a lista se for maior que o comprimento desejado.
    /// </summary>
    /// <param name="tokens">A lista de tokens a ser paddada/truncada.</param>
    /// <param name="length">O comprimento final desejado.</param>
    /// <param name="padValue">O valor do token de padding.</param>
    /// <returns>Uma nova lista de tokens com o comprimento desejado.</returns>
    public static List<int> PadSequence(List<int> tokens, int length, int padValue)
    {
        // Truncar se a sequência for maior que o comprimento
        if (tokens.Count > length)
        {
            return tokens.Take(length).ToList();
        }

        // Adicionar padding se a sequência for menor que o comprimento
        var padded = new List<int>(tokens);
        while (padded.Count < length)
        {
            padded.Add(padValue);
        }
        return padded;
    }

    // Opcional: Método para padar/truncar um tensor 1D
    /// <summary>
    /// Adiciona padding a um tensor 1D ou o trunca para atingir um comprimento específico.
    /// </summary>
    /// <param name="tensor">O tensor 1D.</param>
    /// <param name="length">O comprimento final desejado.</param>
    /// <param name="padValue">O valor do token de padding.</param>
    /// <param name="device">O dispositivo para o novo tensor.</param>
    /// <returns>Um novo tensor 1D com o comprimento desejado, no dispositivo especificado.</returns>
    public static Tensor PadOrTruncateTensor(Tensor tensor, int length, int padValue, Device device)
    {
        if (tensor.dim() != 1)
        {
            throw new ArgumentException("Tensor deve ser 1D.");
        }

        long currentLen = tensor.shape[0];

        if (currentLen > length)
        {
            // Truncar o tensor
            var truncated = tensor.slice(0, 0, length).to(device);
            tensor.Dispose(); // Dispor o tensor original
            return truncated;
        }
        else if (currentLen < length)
        {
            // Criar tensor de padding
            var paddingTensor = full(new long[] { length - currentLen }, padValue, dtype: tensor.dtype, device: device);
            // Concatenar
            var padded = cat(new[] { tensor.to(device), paddingTensor }, dim: 0);
            tensor.Dispose(); // Dispor o tensor original
            paddingTensor.Dispose(); // Dispor o tensor de padding intermediário
            return padded;
        }
        else
        {
            // Nenhum padding/truncamento necessário, apenas mover para o device
            var result = tensor.to(device);
            tensor.Dispose(); // Dispor o tensor original
            return result;
        }
    }
}
// --- END OF FILE PaddingHelper.cs ---