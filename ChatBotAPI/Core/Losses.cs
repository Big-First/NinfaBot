namespace ChatBotAPI.Core;

public static class Losses
{
    public static float BinaryCrossEntropy(Tensor yPred, Tensor yTrue)
    {
        if (!yPred.Shape.SequenceEqual(yTrue.Shape))
            throw new ArgumentException("As formas de yPred e yTrue devem ser iguais.");

        float loss = 0f;
        for (int i = 0; i < yPred.Size; i++)
        {
            float p = yPred._data[i];
            float y = yTrue._data[i];
            // Evitar log(0) com clipping
            p = Math.Max(1e-7f, Math.Min(1 - 1e-7f, p));
            loss += -(y * (float)Math.Log(p) + (1 - y) * (float)Math.Log(1 - p));
        }
        return loss / yPred.Size;
    }
}