using System;
using System.Linq; // Necessário para Any() em verificações futuras talvez
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn; // Para Embedding, Linear

namespace ChatBotAPI.Core
{
    // Herda de nn.Module para usar o ecossistema TorchSharp
    public class TorchSharpModel : Module<Tensor, Tensor>
    {
        // --- Campos NECESSÁRIOS para TorchSharp ---
        private readonly Embedding embedding; // Camada de embedding do TorchSharp
        private readonly Linear linearOutput; // Camada linear do TorchSharp
        private readonly int embeddingSize; // Apenas para referência ou pooling
        private readonly int paddingIdx;    // Índice do token de padding

        public TorchSharpModel(int vocabSize, int embeddingSize, int paddingIdx = 0)
            : base(nameof(TorchSharpModel)) // Nome do módulo
        {
            // Verificações de argumentos
            if (vocabSize <= 0) throw new ArgumentException("Vocab size must be positive", nameof(vocabSize));
            if (embeddingSize <= 0) throw new ArgumentException("Embedding size must be positive", nameof(embeddingSize));

            this.embeddingSize = embeddingSize;
            this.paddingIdx = paddingIdx;

            // --- Definição das Camadas TorchSharp ---
            // paddingIndex diz à camada para não calcular gradientes para este token
            embedding = Embedding(vocabSize, embeddingSize, padding_idx: paddingIdx, this.paddingIdx);
            linearOutput = Linear(embeddingSize, vocabSize);
            // --- Fim da Definição das Camadas ---

            // Registra as camadas para que o TorchSharp as gerencie (parâmetros, save/load, etc.)
            RegisterComponents();

            Console.WriteLine($"TorchSharpModel Initialized: VocabSize={vocabSize}, EmbeddingSize={embeddingSize}, PaddingIdx={this.paddingIdx}");
            // A inicialização dos pesos é feita internamente pelas camadas Linear e Embedding
        }

        // --- Método forward: Define como os dados fluem pelo modelo ---
        public override Tensor forward(Tensor input)
        {
            // Garante tipo Long (acesso a .dtype está correto)
            if (input.dtype != ScalarType.Int64)
            {
                // *** CORREÇÃO: Chama .to() sem parâmetro nomeado ***
                input = input.to(ScalarType.Int64);
            }

            Tensor? embedded = null;
            Tensor? contextVector = null;
            Tensor? logits = null;
            try
            {
                embedded = embedding.forward(input);

                if (input.dim() == 1) // Sem batch
                {
                    // *** CORREÇÃO: Chama .to() sem parâmetro nomeado ***
                    using var mask = input.ne(this.paddingIdx).to(ScalarType.Float32).unsqueeze(1); // [seqLen, 1]
                    Tensor maskedSum = (embedded * mask).sum(dim: 0);

                    using var sumTensorFloat = mask.sum(); // sumTensorFloat é Float32

                    // *** CORREÇÃO: Chama .to() sem parâmetro nomeado ***
                    long nonPaddingCount = sumTensorFloat.to(ScalarType.Int64).item<long>(); // Converte para Int64

                    if (nonPaddingCount > 0) {
                        contextVector = maskedSum / (float)nonPaddingCount;
                    } else {
                        // Acesso a .dtype aqui está correto, parâmetro dtype: em zeros() está correto
                        contextVector = torch.zeros(this.embeddingSize, device: input.device, dtype: embedded.dtype);
                    }
                    maskedSum.Dispose();
                    // sumTensorFloat é descartado pelo using
                }
                else // Com batch
                {
                    contextVector = embedded.mean(new long[] { 1 });
                }

                logits = linearOutput.forward(contextVector);
                return logits;
            }
            finally
            {
                embedded?.Dispose();
                contextVector?.Dispose();
            }
        }

        // --- TODOS os métodos manuais (Train, Predict, Softmax, Attention, DotProduct)
        // --- e campos manuais (embeddings, outputWeights, outputBias)
        // --- foram REMOVIDOS. ---
    }
}