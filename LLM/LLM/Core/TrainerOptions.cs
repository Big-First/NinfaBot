namespace LLM.Core;

public class TrainerOptions
{
    public TrainerOptions(){}
    public int BatchSize { get; set; }
    public int MaxSeqLen { get; set; } // Deve corresponder ou ser menor que o maxSeqLen do modelo
    public int Epochs { get; set; }
    public double LearningRate { get; set; }
    public string SavePath { get; set; }

    public TrainerOptions(int batchSize = default, int maxSeqLen = default, int epochs = default,
        double learningRate = default, string savePath = null)
    {
        BatchSize = batchSize;
        MaxSeqLen = maxSeqLen;
        Epochs = epochs;
        LearningRate = learningRate;
        SavePath = savePath ?? throw new ArgumentNullException(nameof(savePath));
    }
}