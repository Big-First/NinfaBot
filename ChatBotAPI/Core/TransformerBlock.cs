using System;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
namespace ChatBotAPI.Core;

private class TransformerBlock : Module<Tensor, Tensor, Tensor> // Entrada: tensor, mask; Saída: tensor
    {
        private readonly MultiheadAttention attention;
        private readonly LayerNorm norm1, norm2; // Normalização Após Add
        private readonly Sequential feed_forward; // Feed-Forward Network
        private readonly Dropout dropout1, dropout2; // Dropout após atenção e FFN

        /// <summary>
        /// Inicializa um novo bloco Transformer.
        /// </summary>
        /// <param name="embeddingSize">Dimensão dos vetores de embedding e das camadas.</param>
        /// <param name="numHeads">Número de cabeças no mecanismo de Multi-Head Attention.</param>
        /// <param name="dropoutRate">Taxa de dropout a ser aplicada.</param>
        public TransformerBlock(int embeddingSize, int numHeads, double dropoutRate)
            : base(nameof(TransformerBlock))
        {
            // Validações
            if (embeddingSize <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingSize));
            if (numHeads <= 0 || embeddingSize % numHeads != 0) throw new ArgumentOutOfRangeException(nameof(numHeads), "Número de cabeças inválido ou não divide embeddingSize.");
            if (dropoutRate < 0.0 || dropoutRate >= 1.0) throw new ArgumentOutOfRangeException(nameof(dropoutRate));


            // --- Camadas do Bloco ---

            // Self-Attention (Decoder precisa de máscara causal para não "ver" tokens futuros)
            // batch_first=false espera (SeqLen, Batch, EmbeddingSize)
            // A máscara causal será fornecida no método forward.
            this.attention = MultiheadAttention(embeddingSize, numHeads, dropout: dropoutRate, batch_first: false);
            this.norm1 = LayerNorm(embeddingSize); // Normalização após a primeira Add (residual + attention + dropout)
            this.dropout1 = Dropout(dropoutRate); // Dropout aplicado na saída da atenção


            // Feed-Forward Network (FFN)
            // Geralmente, FFN_hidden_size = 4 * embeddingSize (tamanho interno expandido)
            int ffnHiddenSize = embeddingSize * 4;
            this.feed_forward = Sequential(
                Linear(embeddingSize, ffnHiddenSize),
                GELU(), // Função de ativação GELU é comum em Transformers
                Linear(ffnHiddenSize, embeddingSize) // Projeta de volta para embeddingSize
            );
            this.norm2 = LayerNorm(embeddingSize); // Normalização após a segunda Add (residual + ffn + dropout)
            this.dropout2 = Dropout(dropoutRate); // Dropout aplicado na saída da FFN

            // Registra todas as sub-módulos
            RegisterComponents();
            Console.WriteLine($"    TransformerBlock initialized: EmbeddingSize={embeddingSize}, Heads={numHeads}, DropoutRate={dropoutRate}");
        }

        /// <summary>
        /// Realiza o forward pass através de um único bloco Transformer (Decoder).
        /// </summary>
        /// <param name="input">Tensor de entrada para o bloco. Shape: (SeqLen, EmbeddingSize) ou (SeqLen, Batch, EmbeddingSize).</param>
        /// <param name="mask">Máscara de atenção (causal). Shape: (SeqLen, SeqLen) para MultiheadAttention com batch_first=false e máscara booleana.</param>
        /// <returns>Tensor de saída do bloco. Shape: (SeqLen, EmbeddingSize) ou (SeqLen, Batch, EmbeddingSize) - mesmo shape da entrada.</returns>
        public override Tensor forward(Tensor input, Tensor mask) // Recebe a máscara causal
        {
            Tensor? attn_output = null;
            Tensor? attn_output_dropped = null;
            Tensor? residual1 = null;
            Tensor? norm1_output = null; // Saída após a primeira Add&Norm

            Tensor? ffn_output = null;
            Tensor? ffn_output_dropped = null;
            Tensor? residual2 = null;
            Tensor? final_output = null; // Saída final após a segunda Add&Norm


            try
            {
                // Adiciona a dimensão de batch (Batch=1) se o input não tiver (SeqLen, Batch, Features)
                Tensor inputWithBatch;
                if (input.dim() == 2) // Input shape (SeqLen, EmbeddingSize)
                {
                    inputWithBatch = input.unsqueeze(1); // -> (SeqLen, 1, EmbeddingSize)
                }
                else if (input.dim() == 3) // Input shape (SeqLen, Batch, EmbeddingSize) - esperado no treinamento em batch
                {
                    inputWithBatch = input; // Já está no formato correto
                }
                else
                {
                    throw new ArgumentException($"Input tensor has invalid dimension: {input.dim()}. Expected 2 or 3.");
                }


                // --- 1. Self-Attention com Máscara Causal ---
                // MultiheadAttention espera Q, K, V, e a máscara de atenção.
                // Para self-attention, Q, K, V são o mesmo tensor de entrada (inputWithBatch).
                // A máscara causal impede que a atenção veja tokens futuros.
                // O parâmetro attn_mask deve ter shape (SeqLen, SeqLen) para batch_first=false.
                var attnResult = attention.forward(
                    inputWithBatch, inputWithBatch, inputWithBatch, // Q, K, V são os mesmos
                    attn_mask: mask // Passa a máscara causal booleana (SeqLen, SeqLen)
                );
                attn_output = attnResult.Item1; // Saída da atenção. Shape: (SeqLen, Batch, EmbeddingSize)


                // Aplica dropout após atenção
                // O dropout é aplicado na saída da atenção ANTES de adicionar a conexão residual e normalizar.
                attn_output_dropped = dropout1.forward(attn_output);

                // --- Add & Norm 1 (Conexão Residual + Normalização de Camada) ---
                // input (original antes da atenção) + attn_output_dropped
                residual1 = inputWithBatch + attn_output_dropped;
                // Aplica Layer Normalization
                norm1_output = norm1.forward(residual1);


                // --- 2. Feed-Forward Network (FFN) ---
                // Passa a saída normalizada (norm1_output) pela FFN
                ffn_output = feed_forward.forward(norm1_output); // Shape: (SeqLen, Batch, EmbeddingSize)

                // Aplica dropout após FFN
                ffn_output_dropped = dropout2.forward(ffn_output);

                // --- Add & Norm 2 (Conexão Residual + Normalização de Camada) ---
                // norm1_output (residual para a FFN) + ffn_output_dropped
                residual2 = norm1_output + ffn_output_dropped;
                // Aplica Layer Normalization final para este bloco
                final_output = norm2.forward(residual2);

                // Retorna a saída normalizada final do bloco
                // Se o input original não tinha dimensão de batch (dim == 2), removemos a dimensão que adicionamos
                if (input.dim() == 2)
                {
                     return final_output.squeeze(1); // -> (SeqLen, EmbeddingSize)
                }

                return final_output; // Shape (SeqLen, Batch, EmbeddingSize)

            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Error in TransformerBlock.forward: {ex.ToString()}");
                throw;
            }
            finally
            {
                // Dispose dos tensores intermediários
                // IMPORTANTE: inputWithBatch NUNCA deve ser descartado aqui se ele for apenas uma referência
                // para 'input' (se input.dim() == 3). Deve ser descartado APENAS se foi criado via unsqueeze().
                if (input.dim() == 2) inputWithBatch?.Dispose();

                attn_output?.Dispose();
                attn_output_dropped?.Dispose();
                residual1?.Dispose();
                // norm1_output não deve ser descartado imediatamente se for usado na próxima residual connection
                ffn_output?.Dispose();
                ffn_output_dropped?.Dispose();
                residual2?.Dispose();
                // final_output não deve ser descartado, pois é o retorno do método
            }
        }
    } // --- Fim da Classe Interna TransformerBlock ---