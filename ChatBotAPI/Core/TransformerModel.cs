using System;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace AI.Core
{
    // Este é um modelo Decoder-Only Transformer muito simplificado,
    // similar à arquitetura base do GPT, adaptado para a interface Module<Tensor, Tensor>.
    // Ele não inclui todas as otimizações ou recursos de modelos SOTA.
    // CRÍTICO: ESTE MODELO NÃO FOI TREINADO. ELE NÃO FUNCIONARÁ CORRETAMENTE
    // PARA CHAT OU RESUMO SEM TREINAMENTO EXTENSIVO EM DADOS RELEVANTES.

    public class TransformerModel : Module<Tensor, Tensor>
    {
        private readonly Embedding token_embedding;
        private readonly Embedding positional_embedding; // Para Positional Encoding
        private readonly Module<Tensor, Tensor>[] transformer_blocks; // Camadas Transformer
        private readonly LayerNorm final_layer_norm; // Normalização final
        private readonly Linear lm_head; // Camada de saída (Language Model Head)
        private readonly Dropout dropout; // Dropout para regularização

        private readonly int vocabSize;
        private readonly int embeddingSize;
        private readonly int maxSequenceLength;
        private readonly int numHeads;
        private readonly int numLayers;
        private readonly double dropoutRate;

        public TransformerModel(
            int vocabSize,
            int embeddingSize,
            int maxSequenceLength, // Necessário para Positional Encoding
            int numHeads = 4, // Número de cabeças na atenção
            int numLayers = 2, // Número de blocos Transformer
            double dropoutRate = 0.1) // Taxa de dropout
            : base(nameof(TransformerModel))
        {
            // Validações básicas (ajustar conforme necessário)
            if (vocabSize <= 0) throw new ArgumentOutOfRangeException(nameof(vocabSize));
            if (embeddingSize <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingSize));
            if (maxSequenceLength <= 0) throw new ArgumentOutOfRangeException(nameof(maxSequenceLength));
            if (numHeads <= 0 || embeddingSize % numHeads != 0) throw new ArgumentOutOfRangeException(nameof(numHeads), "Número de cabeças inválido ou não divide embeddingSize.");
            if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
            if (dropoutRate < 0.0 || dropoutRate >= 1.0) throw new ArgumentOutOfRangeException(nameof(dropoutRate));

            this.vocabSize = vocabSize;
            this.embeddingSize = embeddingSize;
            this.maxSequenceLength = maxSequenceLength;
            this.numHeads = numHeads;
            this.numLayers = numLayers;
            this.dropoutRate = dropoutRate;


            // --- Definição das Camadas ---
            this.token_embedding = Embedding(vocabSize, embeddingSize);
            // Positional Encoding: Aprende embeddings para cada posição até maxSequenceLength
            this.positional_embedding = Embedding(maxSequenceLength, embeddingSize);
            this.dropout = Dropout(dropoutRate);

            // Blocos Transformer (Decoder Layers)
            this.transformer_blocks = new Module<Tensor, Tensor>[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                this.transformer_blocks[i] = new TransformerBlock(embeddingSize, numHeads, dropoutRate);
            }

            this.final_layer_norm = LayerNorm(embeddingSize); // Normalização antes da saída
            this.lm_head = Linear(embeddingSize, vocabSize); // Mapeia para o espaço do vocabulário

            RegisterComponents(); // Registra todas as camadas para gerenciamento

            Console.WriteLine($"TransformerModel (Decoder-Only) Initialized: VocabSize={vocabSize}, EmbeddingSize={embeddingSize}, MaxSeqLen={maxSequenceLength}, Heads={numHeads}, Layers={numLayers}, DropoutRate={dropoutRate}");
        }

        /// <summary>
        /// Realiza o forward pass através do modelo Transformer.
        /// </summary>
        /// <param name="input">Input tensor contendo índices de token. Shape: (SeqLen).</param>
        /// <returns>Output tensor contendo logits para a previsão do próximo token. Shape: (VocabSize).</returns>
        /// <remarks>
        /// O input é esperado como uma sequência 1D de IDs de token.
        /// O modelo processa a sequência e retorna os logits para o último token na sequência,
        /// prevendo o próximo token.
        /// </remarks>
        public override Tensor forward(Tensor input)
        {
            Tensor? tokenEmbeddings = null;
            Tensor? positionalEmbeddings = null;
            Tensor? x = null; // Tensor que passa pelos blocos Transformer
            Tensor? finalOutput = null; // Saída após LayerNorm

            try
            {
                // Garante tipo Long
                if (input.dtype != ScalarType.Int64) { input = input.to(ScalarType.Int64); }

                int seqLen = (int)input.shape[0];
                 // CRÍTICO: Valida o comprimento da sequência
                if (seqLen > this.maxSequenceLength) {
                     // Isso NÃO deve acontecer se o input já foi truncado ANTES de chamar forward
                     // Mas é uma salvaguarda. Idealmente, o truncamento ocorre no nível do caller (Program.cs/HandleWebSocketAsync)
                     Console.Error.WriteLine($"Warning: Input sequence length ({seqLen}) exceeds model's max_seq_len ({this.maxSequenceLength}). Truncating inside forward.");
                     input = input.slice(0, seqLen - this.maxSequenceLength, seqLen);
                     seqLen = this.maxSequenceLength;
                }


                // 1. Embeddings (Token + Posicional)
                tokenEmbeddings = token_embedding.forward(input); // Shape: (SeqLen, EmbeddingSize)

                // Cria tensor de posições (0, 1, 2, ..., seqLen-1)
                using var positions = torch.arange(seqLen, dtype: ScalarType.Int64).to(input.device);
                positionalEmbeddings = positional_embedding.forward(positions); // Shape: (SeqLen, EmbeddingSize)

                // Combina embeddings
                x = tokenEmbeddings + positionalEmbeddings;

                // Aplica dropout inicial
                x = dropout.forward(x);


                // 2. Passar pelos Blocos Transformer
                // Como é Decoder-Only, precisamos de uma máscara causal (look-ahead mask)
                // para que cada token na sequência de saída só possa atender a tokens anteriores.
                // Esta máscara é uma matriz triangular inferior.
                // Shape da máscara: (SeqLen, SeqLen). Elementos true/1 bloqueiam a atenção.
                using var causalMask = torch.triu(torch.ones(seqLen, seqLen, dtype: ScalarType.Bool), diagonal: 1).to(input.device);


                for (int i = 0; i < numLayers; i++)
                {
                    // Cada bloco Transformer recebe a entrada atual (x) e a máscara causal
                    x = ((TransformerBlock)transformer_blocks[i]).forward(x, causalMask);
                    // Verifica nulidade após cada bloco (opcional, para debug)
                    if ((bool)(x == null)) throw new NullReferenceException($"Output of TransformerBlock {i} is null.");
                }

                // 3. Normalização Final
                finalOutput = final_layer_norm.forward(x);

                // 4. Camada Linear de Saída (LM Head)
                // Precisamos dos logits APENAS para o último token na sequência de saída,
                // pois estamos prevendo o PRÓXIMO token.
                // last_token_output shape: (EmbeddingSize)
                using var lastTokenOutput = finalOutput.select(0, -1); // Seleciona o último elemento na dimensão de sequência (dim 0)

                // logits shape: (VocabSize)
                var logits = lm_head.forward(lastTokenOutput);

                // Verifica se logits são válidos
                if ((bool)(logits == null))
                {
                    throw new InvalidOperationException("Logits became null after linear layer.");
                }

                // 5. RETORNAR OS LOGITS
                return logits; // Shape: (VocabSize)

            }
            catch (Exception ex)
            {
                 Console.Error.WriteLine($"Error in TransformerModel.forward: {ex.ToString()}");
                 throw; // Relança a exceção
            }
            finally
            {
                // Dispose dos tensores intermediários (garantir que todos sejam descartados)
                tokenEmbeddings?.Dispose();
                positionalEmbeddings?.Dispose();
                // 'x' não pode ser descartado aqui, pois ele é reatribuído no loop.
                // Mas as saídas dos blocos Transformer são consumidas na próxima iteração,
                // então o GC do TorchSharp deve eventualmente liberá-los se não houver vazamentos.
                // Descarte da máscara causal
                // causalMask?.Dispose(); // Disposable via `using` block
                finalOutput?.Dispose();
                // lastTokenOutput Disposable via `using` block
                // 'logits' é o tensor de retorno e não deve ser descartado aqui.
            }
        }

        // --- Definição de Classes Internas para Blocos do Transformer ---

        // Representa um bloco Decoder simples (Self-Attention + FeedForward)
        private class TransformerBlock : Module<Tensor, Tensor, Tensor> // Entrada: tensor, mask; Saída: tensor
        {
            private readonly MultiheadAttention attention;
            private readonly LayerNorm norm1, norm2;
            private readonly Sequential feed_forward;
            private readonly Dropout dropout1, dropout2;

            public TransformerBlock(int embeddingSize, int numHeads, double dropoutRate)
                : base(nameof(TransformerBlock))
            {
                // Self-Attention (Decoder precisa de máscara causal)
                // batch_first=false espera (SeqLen, Batch, EmbeddingSize)
                this.attention = MultiheadAttention(embeddingSize, numHeads, dropout: dropoutRate, batch_first: false);
                this.norm1 = LayerNorm(embeddingSize);
                this.dropout1 = Dropout(dropoutRate);

                // Feed-Forward Network (FFN)
                // Geralmente, FFN_hidden_size = 4 * embeddingSize
                int ffnHiddenSize = embeddingSize * 4;
                this.feed_forward = Sequential(
                    Linear(embeddingSize, ffnHiddenSize),
                    GELU(), // Função de ativação comum em Transformers
                    Linear(ffnHiddenSize, embeddingSize)
                );
                this.norm2 = LayerNorm(embeddingSize);
                this.dropout2 = Dropout(dropoutRate);

                RegisterComponents();
            }

            // forward: input shape (SeqLen, EmbeddingSize), mask shape (SeqLen, SeqLen)
            public override Tensor forward(Tensor input, Tensor mask) // Recebe a máscara causal
            {
                Tensor? attn_output = null;
                Tensor? attn_output_dropped = null;
                Tensor? residual1 = null;
                Tensor? norm1_output = null;

                Tensor? ffn_output = null;
                Tensor? ffn_output_dropped = null;
                Tensor? residual2 = null;
                Tensor? norm2_output = null;


                try
                {
                    // --- 1. Self-Attention ---
                    // Adiciona a dimensão de batch (SeqLen, 1, EmbeddingSize) para o MultiheadAttention
                    using var inputWithBatch = input.unsqueeze(1);

                    // Mask: A máscara deve ser (Batch, NumHeads, SeqLen, SeqLen) para MultiheadAttention com batch_first=false
                    // Ou (SeqLen, SeqLen) se for uma máscara aditiva (com valores muito negativos nos locais bloqueados)
                    // Ou simplesmente (SeqLen, SeqLen) para o parâmetro `attn_mask` se for booleana ou float para máscara aditiva.
                    // Vamos usar a máscara booleana (SeqLen, SeqLen) que criamos no forward principal.
                    var attnResult = attention.forward(
                        inputWithBatch, inputWithBatch, inputWithBatch, // Q, K, V são os mesmos para self-attention
                        attn_mask: mask // Passa a máscara causal booleana
                    );
                    attn_output = attnResult.Item1; // Saída da atenção. Shape: (SeqLen, 1, EmbeddingSize)

                    // Remove a dimensão de batch de volta
                    attn_output = attn_output.squeeze(1); // Shape: (SeqLen, EmbeddingSize)

                    // Aplica dropout após atenção
                    attn_output_dropped = dropout1.forward(attn_output);

                    // --- Add & Norm 1 (ResNet + LayerNorm) ---
                    // Soma a entrada original (residual connection) com a saída da atenção + dropout
                    residual1 = input + attn_output_dropped;
                    // Aplica Layer Normalization
                    norm1_output = norm1.forward(residual1);


                    // --- 2. Feed-Forward Network ---
                    ffn_output = feed_forward.forward(norm1_output); // Shape: (SeqLen, EmbeddingSize)

                    // Aplica dropout após FFN
                    ffn_output_dropped = dropout2.forward(ffn_output);

                    // --- Add & Norm 2 (ResNet + LayerNorm) ---
                    // Soma a saída da LayerNorm 1 (residual connection para a FFN) com a saída da FFN + dropout
                    residual2 = norm1_output + ffn_output_dropped;
                    // Aplica Layer Normalization final para este bloco
                    norm2_output = norm2.forward(residual2);

                    // Retorna a saída normalizada
                    return norm2_output;

                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"Error in TransformerBlock.forward: {ex.ToString()}");
                    throw;
                }
                finally
                {
                    // Dispose dos tensores intermediários
                    attn_output?.Dispose();
                    attn_output_dropped?.Dispose();
                    residual1?.Dispose();
                    norm1_output?.Dispose(); // Este tensor também é a entrada para o próximo passo dentro do bloco FFN, cuidado
                    ffn_output?.Dispose();
                    ffn_output_dropped?.Dispose();
                    residual2?.Dispose();
                    // norm2_output não deve ser descartado, pois é o retorno do método
                }
            }
        }
    }
}