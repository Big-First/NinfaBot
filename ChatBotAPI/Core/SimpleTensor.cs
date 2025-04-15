namespace ChatBotAPI.Core;

public class SimpleTensor<T> where T : struct
{
    public T[] Data { get; }
    public int[] Shape { get; }

    public SimpleTensor(T[] data, int[] shape)
    {
        Data = data;
        Shape = shape;
        if (data.Length != shape.Aggregate((a, b) => a * b))
            throw new ArgumentException("Dados não correspondem ao shape.");
    }

    public T this[int batch, int seq]
    {
        get => Data[batch * Shape[1] + seq];
        set => Data[batch * Shape[1] + seq] = value;
    }
}