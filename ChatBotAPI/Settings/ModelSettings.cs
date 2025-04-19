namespace ChatBotAPI.Settings
{

    public class ModelSettings
    {
        public string TokenizerConfigPath { get; set; } = string.Empty;
        public int MaxSequenceLength { get; set; } = 50;
        public int EmbeddingSize { get; set; } = 128;
        public int VocabSizeLimit { get; set; } = 60000;
        public int MaxTreeDepth { get; set; } = 5;
        public int TrainingEpochs { get; set; } = 10;
        public string ModelSavePath { get; set; }
        public float SamplingTemperature { get; set; } = 0.7f;
        public int TopK { get; set; } = 0;
        public float TopP { get; set; } = 0.9f;
    }
}