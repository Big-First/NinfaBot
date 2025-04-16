namespace ChatBotAPI.Settings
{

    public class ModelSettings
    {
        public string TokenizerConfigPath { get; set; } = string.Empty;
        public int MaxSequenceLength { get; set; } = 50;
        public int EmbeddingSize { get; set; } = 128;
        public int VocabSizeLimit { get; set; } = 10000;
        public int MaxTreeDepth { get; set; } = 5;
        public int TrainingEpochs { get; set; } = 10;
    }
}