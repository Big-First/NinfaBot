// Core/PositionalEncoding.cs - Alternativa com Embedding Aprendível

using System;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ChatBotAPI.Core
{
    public class PositionalEncoding : Module<Tensor, Tensor>
    {
        private readonly Embedding pos_embedding;
        private readonly Dropout dropout; // <--- Dropout já existe
        private readonly int maxLen;

        public PositionalEncoding(int dModel, double dropoutRate, int maxLen = 5000)
            : base(nameof(PositionalEncoding))
        {
            // ...
            this.dropout = Dropout(dropoutRate); // <--- Inicializado aqui
            this.pos_embedding = Embedding(maxLen, dModel);
            RegisterComponents(); // Registra dropout e pos_embedding
            Console.WriteLine($"PositionalEncoding: Initialized with learnable embeddings and Dropout={dropoutRate}.");
        }

        public override Tensor forward(Tensor x)
        {
            // ... (cria posições, obtém pos_enc) ...
            Device currentDevice = x.device;
            long seqLen = x.shape[0];
            long batchSize = x.shape[1];
            Tensor? pos_enc = null;
            Tensor? result = null;
            Tensor? dropout_output = null;
            try {
                using var positions_base = torch.arange(0, seqLen, ScalarType.Int64, currentDevice);
                using var positions_expanded = positions_base.unsqueeze(1).expand(seqLen, batchSize);
                pos_enc = this.pos_embedding.forward(positions_expanded);
                result = x + pos_enc;
                dropout_output = this.dropout.forward(result); // <--- Aplicado aqui
                return dropout_output;
            } finally {
                pos_enc?.Dispose();
                result?.Dispose();
                // dropout_output é retornado
            }
        }
    }
}