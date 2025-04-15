using System;
using System.Linq;

namespace ChatBotAPI.Core;

// Funções de ativação e perda
public static class Activations
{
    public static Tensor ReLU(Tensor x)
    {
        var result = new Tensor(x.Shape.ToArray());
        for (int i = 0; i < x.Size; i++)
            result._data[i] = Math.Max(0, x._data[i]);
        return result;
    }

    public static Tensor Sigmoid(Tensor x)
    {
        var result = new Tensor(x.Shape.ToArray());
        for (int i = 0; i < x.Size; i++)
            result._data[i] = (float)(1.0 / (1.0 + Math.Exp(-x._data[i])));
        return result;
    }
}