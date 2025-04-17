// TorchSharpModel.cs - MODIFICADO COM LSTM

using System;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules; // Necessário para LSTM e outras camadas nn.Module
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ChatBotAPI.Core
{
    public class TorchSharpModel : Module<Tensor, Tensor> // Assinatura não muda
    {
        private readonly Embedding embedding;
        private readonly LSTM lstm; // Nova camada LSTM
        private readonly Linear linearOutput;
        private readonly int hiddenSize; // Tamanho do estado oculto da LSTM
        private readonly int paddingIdx;

        // Construtor modificado para incluir LSTM
        public TorchSharpModel(int vocabSize, int embeddingSize, int paddingIdx = 0, int hiddenSize = 128, int numLSTMLayers = 1) // hiddenSize pode ser igual a embeddingSize ou diferente
            : base(nameof(TorchSharpModel))
        {
            if (vocabSize <= 0) throw new ArgumentException("Vocab size must be positive", nameof(vocabSize));
            if (embeddingSize <= 0) throw new ArgumentException("Embedding size must be positive", nameof(embeddingSize));
            if (hiddenSize <= 0) throw new ArgumentException("Hidden size must be positive", nameof(hiddenSize));
            if (numLSTMLayers <= 0) throw new ArgumentException("Number of LSTM layers must be positive", nameof(numLSTMLayers));


            this.hiddenSize = hiddenSize; // Guarda o hiddenSize
            this.paddingIdx = paddingIdx;

            // --- Definição das Camadas ---
            // 1. Embedding (como antes, usa padding_idx)
            embedding = Embedding(vocabSize, embeddingSize, padding_idx: this.paddingIdx);

            // 2. LSTM
            // input_size: tamanho do vetor de entrada em cada passo (vem do embedding) = embeddingSize
            // hidden_size: tamanho do estado oculto e do estado da célula = hiddenSize (parâmetro do construtor)
            // num_layers: número de camadas LSTM empilhadas (1 para começar)
            // batch_first: False (padrão) -> input deve ser (seq_len, batch_size, input_size)
            lstm = LSTM(inputSize: embeddingSize, // <- Correção
                hiddenSize: this.hiddenSize, // <- Correção (já estava camelCase, mas confirmando)
                numLayers: numLSTMLayers,  // <- Correção
                batchFirst: false);

            // 3. Linear Output (agora recebe hidden_size da LSTM)
            linearOutput = Linear(inputSize: this.hiddenSize, // <- Correção
                outputSize: vocabSize); 
            // --- Fim Definição ---

            // Registra todas as camadas
            RegisterComponents();

            Console.WriteLine($"TorchSharpModel (LSTM) Initialized: VocabSize={vocabSize}, EmbeddingSize={embeddingSize}, HiddenSize={this.hiddenSize}, LSTMLayers={numLSTMLayers}, PaddingIdx={this.paddingIdx}");
        }

        // --- Método forward MODIFICADO para usar LSTM ---
        public override Tensor forward(Tensor input)
        {
            // ... (Garante tipo Long) ...

            Tensor? embedded = null;
            Tensor? lstmInput = null;
            // Variáveis para receber o retorno completo da LSTM
            Tensor? lstmOutputSequence = null;
            Tensor? h_n = null; // Último estado oculto
            Tensor? c_n = null; // Último estado da célula
            // Variáveis para o resto
            Tensor? lastOutput = null;
            Tensor? logits = null;

            try
            {
                embedded = embedding.forward(input);
                lstmInput = embedded.unsqueeze(1); // Shape: (seq_len, 1, embedding_size)

                // --- CORREÇÃO NA DESCONSTRUÇÃO ---
                // 1. Chama forward e armazena o resultado da tupla completa
                Console.WriteLine($"DEBUG LSTM Forward: Input shape {lstmInput.shape}");
                var lstmResult = lstm.forward(lstmInput);

                // 2. Desconstrói a tupla armazenada
                lstmOutputSequence = lstmResult.output; // A saída da sequência
                (h_n, c_n) = lstmResult.hidden;       // A tupla de estados finais

                 Console.WriteLine($"DEBUG LSTM Forward: Output sequence shape {lstmOutputSequence.shape}");
                 // Você pode adicionar logs opcionais para h_n.shape e c_n.shape se quiser

                // 4. Obter a Saída Relevante (último passo)
                lastOutput = lstmOutputSequence[-1, 0, ..];
                 Console.WriteLine($"DEBUG LSTM Forward: Last LSTM output shape {lastOutput.shape}");

                // 5. Passar pela Camada Linear Final
                logits = linearOutput.forward(lastOutput);
                 Console.WriteLine($"DEBUG LSTM Forward: Final logits shape {logits.shape}");

                return logits;
            }
            catch(Exception ex) { /* ... Bloco catch como antes ... */ }
            finally
            {
                // Descarta tensores intermediários (incluindo os da tupla LSTM)
                embedded?.Dispose();
                lstmInput?.Dispose();
                lstmOutputSequence?.Dispose();
                h_n?.Dispose();
                c_n?.Dispose();
                lastOutput?.Dispose();
            }
        }
    }
}