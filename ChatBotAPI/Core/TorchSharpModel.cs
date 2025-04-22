using System;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules; // Necessário para Dropout
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace AI.Core
{
    /// <summary>
    /// Implements a neural network model using TorchSharp with LSTM architecture and dropout regularization.
    /// This model is designed for sequence processing tasks, particularly text generation or classification.
    /// </summary>
    public class TorchSharpModel : Module<Tensor, Tensor>
    {
        /// <summary>
        /// Embedding layer that converts token indices to dense vectors.
        /// </summary>
        private readonly Embedding embedding;

        /// <summary>
        /// LSTM layer for processing sequential data.
        /// </summary>
        private readonly LSTM lstm;

        /// <summary>
        /// Dropout layer for regularization to prevent overfitting.
        /// </summary>
        private readonly Dropout lstm_dropout;

        /// <summary>
        /// Linear layer for final output transformation.
        /// </summary>
        private readonly Linear linearOutput;

        /// <summary>
        /// Size of the hidden state in the LSTM layer.
        /// </summary>
        private readonly int hiddenSize;

        /// <summary>
        /// Index used for padding tokens in the vocabulary.
        /// </summary>
        private readonly int paddingIdx;

        /// <summary>
        /// Number of LSTM layers in the model.
        /// </summary>
        private readonly int numLSTMLayers;

        /// <summary>
        /// Initializes a new instance of the TorchSharpModel with specified parameters.
        /// </summary>
        /// <param name="vocabSize">Size of the vocabulary (number of unique tokens).</param>
        /// <param name="embeddingSize">Dimension of the embedding vectors.</param>
        /// <param name="paddingIdx">Index used for padding tokens (default: 0).</param>
        /// <param name="hiddenSize">Size of the LSTM hidden state (default: 128).</param>
        /// <param name="numLSTMLayers">Number of LSTM layers (default: 1).</param>
        /// <param name="dropoutRate">Probability of dropout for regularization (default: 0.2).</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when any parameter is invalid.</exception>
        public TorchSharpModel(
            int vocabSize,
            int embeddingSize,
            int paddingIdx = 0,
            int hiddenSize = 128,
            int numLSTMLayers = 1,
            double dropoutRate = 0.2)
            : base(nameof(TorchSharpModel))
        {
            // Validações básicas
            if (vocabSize <= 0) throw new ArgumentOutOfRangeException(nameof(vocabSize));
            if (embeddingSize <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingSize));
            if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
            if (numLSTMLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLSTMLayers));
            if (dropoutRate < 0.0 || dropoutRate >= 1.0) throw new ArgumentOutOfRangeException(nameof(dropoutRate)); // Dropout é probabilidade [0, 1)

            this.hiddenSize = hiddenSize;
            this.paddingIdx = paddingIdx;
            this.numLSTMLayers = numLSTMLayers;

            // Definição das Camadas
            this.embedding = Embedding(vocabSize, embeddingSize, padding_idx: this.paddingIdx);
            this.lstm = LSTM(inputSize: embeddingSize, hiddenSize: this.hiddenSize, numLayers: this.numLSTMLayers, batchFirst: false); // batchFirst=false espera (SeqLen, Batch, Features)
            this.lstm_dropout = Dropout(p: dropoutRate);
            this.linearOutput = Linear(inputSize: this.hiddenSize, outputSize: vocabSize);

            RegisterComponents(); // Registra todas as camadas (incluindo dropout) para gerenciamento

            Console.WriteLine($"TorchSharpModel (LSTM) Initialized: VocabSize={vocabSize}, EmbeddingSize={embeddingSize}, HiddenSize={this.hiddenSize}, LSTMLayers={this.numLSTMLayers}, PaddingIdx={this.paddingIdx}, DropoutRate={dropoutRate}"); // Log atualizado
        }

        /// <summary>
        /// Performs the forward pass through the neural network.
        /// </summary>
        /// <param name="input">Input tensor containing token indices.</param>
        /// <returns>Output tensor containing logits for the next token prediction.</returns>
        /// <exception cref="NullReferenceException">Thrown when LSTM output sequence is null.</exception>
        /// <exception cref="InvalidOperationException">Thrown when logits become null after linear layer.</exception>
        public override Tensor forward(Tensor input)
        {
            Tensor? embedded = null;
            Tensor? lstmInput = null;
            Tensor? lstmOutputSequence = null;
            Tensor? h_n = null; // Apenas estado oculto final (da tentativa anterior)
            Tensor? stateTupleObj = null; // Para guardar o objeto de estado completo
            Tensor? lastTimeStepHiddenState = null;
            Tensor? dropoutOutput = null; // *** NOVO: Saída do Dropout ***
            Tensor? logits = null;

            try
            {
                // 1. Garante tipo Long
                if (input.dtype != ScalarType.Int64) { input = input.to(ScalarType.Int64); }

                // 2. Embedding
                embedded = embedding.forward(input); // Shape: (SeqLen, EmbeddingSize) para batch_size=1

                // 3. Ajustar Shape para LSTM (SeqLen, Batch=1, Features)
                // O LSTM espera (SeqLen, BatchSize, InputSize) por padrão (batchFirst=false)
                // Se o input já tem shape (SeqLen), unsqueeze(1) cria (SeqLen, 1, EmbeddingSize)
                lstmInput = embedded.unsqueeze(1);

                // 4. Passar pela LSTM
                // Retorna (outputSeq[SeqLen, Batch, HiddenSize], stateTuple[h_n, c_n])
                // Onde h_n e c_n têm shape (NumLayers, Batch, HiddenSize)
                var lstmResult = lstm.forward(lstmInput);
                lstmOutputSequence = lstmResult.Item1;
                stateTupleObj = lstmResult.Item2 as Tensor; // Tenta pegar o estado (pode ser tupla) - MANTIDO POR ENQUANTO, MAS PROVAVELMENTE NÃO É USADO DIRETAMENTE

                // Verifica se a sequência de saída é válida
                if ((bool)(lstmOutputSequence == null)) // Usar comparação direta de referência
                {
                     Console.Error.WriteLine("CRITICAL ERROR: LSTM output sequence (Item1) is null!");
                     throw new NullReferenceException("LSTM output sequence is null.");
                }

                // 5. Obter a Saída Relevante (do último passo de tempo da sequência)
                // outputSequence tem shape (SeqLen, Batch=1, HiddenSize)
                // select(0, -1) pega o último item na dimensão 0 (tempo) -> Shape (Batch=1, HiddenSize)
                // squeeze(0) remove a dimensão do batch -> Shape (HiddenSize)
                lastTimeStepHiddenState = lstmOutputSequence.select(0, -1).squeeze(0);

                // *** NOVO: 6. Aplicar Dropout ***
                // Dropout é aplicado apenas durante model.train()
                dropoutOutput = lstm_dropout.forward(lastTimeStepHiddenState);

                 // Verifica se a saída do dropout é válida
                 if ((bool)(dropoutOutput == null)) {
                      Console.Error.WriteLine("CRITICAL ERROR: Dropout output is null!");
                      throw new NullReferenceException("Dropout output is null.");
                 }

                // 7. Passar pela Camada Linear Final (usando a saída do dropout)
                logits = linearOutput.forward(dropoutOutput);

                // Verifica se logits são válidos
                if ((bool)(logits == null)) // Usa comparação direta
                {
                    throw new InvalidOperationException("Logits became null after linear layer.");
                }

                // 8. RETORNAR OS LOGITS
                return logits; // Shape esperado: (VocabSize)

            }
            catch (Exception ex)
            {
                 Console.Error.WriteLine($"Error in TorchSharpModel.forward: {ex.ToString()}"); // Log detalhado
                 // Considerar relançar ou retornar um tensor inválido/nulo controlado
                 throw; // Relança a exceção por padrão
            }
            finally
            {
                // Dispose dos tensores intermediários
                embedded?.Dispose();
                lstmInput?.Dispose();
                lstmOutputSequence?.Dispose();
                h_n?.Dispose(); // Mesmo que o cast possa falhar, tenta descartar se não for nulo
                // Descarta o stateTupleObj se necessário (depende do tipo real)
                (stateTupleObj as IDisposable)?.Dispose();
                lastTimeStepHiddenState?.Dispose();
                dropoutOutput?.Dispose(); // *** NOVO: Dispose da saída do Dropout ***
            }
        } // Fim forward
    }
}