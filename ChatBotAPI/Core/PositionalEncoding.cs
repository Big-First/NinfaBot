// Core/PositionalEncoding.cs - Alternativa com Embedding Aprendível

using System;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ChatBotAPI.Core
{
    public class PositionalEncoding : Module<Tensor, Tensor> // Mantém a mesma interface
    {
        private readonly Embedding pos_embedding;
        private readonly Dropout dropout;
        private readonly int maxLen;
        private Device _device = torch.CPU; // Guarda o device

        public PositionalEncoding(int dModel, double dropoutRate, int maxLen = 5000)
            : base(nameof(PositionalEncoding))
        {
            if (maxLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxLen), "maxLen must be positive.");

            this.maxLen = maxLen;
            this.dropout = Dropout(dropoutRate);

            // Cria uma camada de Embedding:
            // - num_embeddings: O número máximo de posições (maxLen)
            // - embedding_dim: A dimensão do modelo (dModel)
            this.pos_embedding = Embedding(maxLen, dModel);
            // Os pesos deste embedding serão aprendidos durante o treinamento

            RegisterComponents(); // Registra dropout e pos_embedding
            Console.WriteLine($"PositionalEncoding: Using LEARNABLE embeddings (maxLen={maxLen}, dModel={dModel}).");
        }

        public override Tensor forward(Tensor x)
        {
            // x shape: (SeqLen, BatchSize, dModel)

            long seqLen = x.shape[0];
            long batchSize = x.shape[1]; // Precisamos do batch size

            if (seqLen > this.maxLen)
            {
                 throw new ArgumentOutOfRangeException(nameof(x), $"Input sequence length ({seqLen}) exceeds PositionalEncoding maxLen ({this.maxLen})");
            }

            // Cria um tensor de posições: 0, 1, 2, ..., seqLen-1
            // Precisamos expandir para o batch size.
            // Shape: (SeqLen)
            using var positions_base = torch.arange(0, seqLen, ScalarType.Int64, this._device);

            // Expande para (SeqLen, BatchSize) replicando as posições para cada item no batch
            // Shape: (SeqLen, BatchSize)
             using var positions_expanded = positions_base.unsqueeze(1).expand(seqLen, batchSize);

            // Obtém os embeddings posicionais
            // Input para embedding: (SeqLen, BatchSize)
            // Output: (SeqLen, BatchSize, dModel)
             using var pos_enc = this.pos_embedding.forward(positions_expanded);

            // Soma os embeddings posicionais aos embeddings dos tokens
            // (SeqLen, BatchSize, dModel) + (SeqLen, BatchSize, dModel)
             var result = x + pos_enc;

            return this.dropout.forward(result);

             // Gerenciamento de Dispose implícito com using var onde possível
        }
    }
}