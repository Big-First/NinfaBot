using ChatBotAPI.enums;

namespace ChatBotAPI.Settings
{

    public class ModelSettings
    {
        // --- Configurações Antigas ---
        // public string TokenizerConfigPath { get; set; } = string.Empty; // Não mais necessário para SharpToken
        // public string MergesPath { get; set; } = string.Empty; // Não mais necessário para SharpToken
        public int MaxSequenceLength { get; set; } = 50;
        // public int EmbeddingSize { get; set; } = 128; // Substituído por DModel
        public string ModelSavePath { get; set; } = "model_transformer_state.pt"; // Novo nome de arquivo
        public int TrainingEpochs { get; set; } = 10;
        public float SamplingTemperature { get; set; } = 0.7f;
        public int TopK { get; set; } = 40; // Ajustado valor padrão
        public float TopP { get; set; } = 0.9f;
        public DecodingStrategy DecodingStrategy { get; set; } = DecodingStrategy.Sampling;

        // --- NOVOS Hiperparâmetros do Transformer ---
        public int DModel { get; set; } = 256;       // Dimensão do Embedding e do Modelo (Ex: 256, 512, 768)
        public int Nhead { get; set; } = 4;         // Número de Cabeças de Atenção (deve dividir DModel) (Ex: 4, 8)
        public int NumDecoderLayers { get; set; } = 3; // Número de Camadas Decoder (Ex: 3, 6)
        public int DimFeedforward { get; set; } = 512; // Dimensão da Camada FeedForward interna (Ex: 512, 1024, 2048)
        public double DropoutRate { get; set; } = 0.1; // Taxa de Dropout

        // public int VocabSizeLimit { get; set; } = 60000; // Não necessário com SharpToken
        // public int MaxTreeDepth { get; set; } = 5;       // Não relevante para Transformer
    }
}