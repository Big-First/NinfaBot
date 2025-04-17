// TorchSharpModel.cs - VERSÃO LSTM ASSUMINDO Item2 É APENAS h_n

using System;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ChatBotAPI.Core
{
    public class TorchSharpModel : Module<Tensor, Tensor>
    {
        private readonly Embedding embedding;
        private readonly LSTM lstm;
        private readonly Linear linearOutput;
        private readonly int hiddenSize;
        private readonly int paddingIdx;
        private readonly int numLSTMLayers;

        // Construtor (Correto)
        public TorchSharpModel(int vocabSize, int embeddingSize, int paddingIdx = 0, int hiddenSize = 128, int numLSTMLayers = 1)
            : base(nameof(TorchSharpModel))
        {
            // ... (Validações, Definição Camadas, RegisterComponents, Log) ...
             this.hiddenSize = hiddenSize;
            this.paddingIdx = paddingIdx;
            this.numLSTMLayers = numLSTMLayers;
            this.embedding = Embedding(vocabSize, embeddingSize, padding_idx: this.paddingIdx);
            this.lstm = LSTM(inputSize: embeddingSize, hiddenSize: this.hiddenSize, numLayers: this.numLSTMLayers, batchFirst: false);
            this.linearOutput = Linear(inputSize: this.hiddenSize, outputSize: vocabSize);
            RegisterComponents();
            Console.WriteLine($"TorchSharpModel (LSTM) Initialized: VocabSize={vocabSize}, EmbeddingSize={embeddingSize}, HiddenSize={this.hiddenSize}, LSTMLayers={this.numLSTMLayers}, PaddingIdx={this.paddingIdx}");
        }

        // --- Método forward com LSTM ---
        public override Tensor forward(Tensor input)
        {
            Tensor? embedded = null;
            Tensor? lstmInput = null;
            Tensor? lstmOutputSequence = null;
            Tensor? h_n = null; // Apenas estado oculto final
            Tensor? lastTimeStepHiddenState = null;
            Tensor? logits = null;

            try
            {
                // 1. Garante tipo Long
                if (input.dtype != ScalarType.Int64) { input = input.to(ScalarType.Int64); }

                // 2. Embedding
                embedded = embedding.forward(input);

                // 3. Ajustar Shape para LSTM
                lstmInput = embedded.unsqueeze(1);

                // 4. Passar pela LSTM
                var lstmResult = lstm.forward(lstmInput); // Retorna (Tensor outputSeq, object state)

                // 5. Extrai output e TENTA pegar h_n de Item2
                lstmOutputSequence = lstmResult.Item1;
                h_n = lstmResult.Item2 as Tensor; // Tenta cast direto para Tensor

                // Verifica se a sequência de saída ou o estado oculto são nulos
                if ((bool)(lstmOutputSequence == null)) {
                     Console.Error.WriteLine("CRITICAL ERROR: LSTM output sequence (Item1) is null!");
                     throw new NullReferenceException("LSTM output sequence is null.");
                }
                if ((bool)(h_n == null)) {
                     // Se o cast direto 'as Tensor' falhou, Item2 NÃO é um Tensor simples.
                     // Não conseguimos determinar o tipo exato de forma segura sem Reflection ou mais informações.
                     // Lançamos um erro claro.
                     Console.Error.WriteLine($"CRITICAL ERROR: LSTM final state (Item2) is not a Tensor or could not be cast. Type is {lstmResult.Item2?.GetType().FullName}. Cannot reliably get h_n.");
                     throw new InvalidCastException($"Cannot cast LSTM state (Item2) to Tensor. Actual type: {lstmResult.Item2?.GetType().FullName}");
                }
                // Se chegou aqui, temos lstmOutputSequence e h_n (ambos não nulos)

                // Console.WriteLine($"DEBUG LSTM Forward: Output sequence shape {lstmOutputSequence.shape}");
                // Console.WriteLine($"DEBUG LSTM Forward: Final Hidden (h_n) Shape {h_n.shape}");

                // 6. Obter a Saída Relevante
                 lastTimeStepHiddenState = lstmOutputSequence.select(0, -1).squeeze(0);
                // Console.WriteLine($"DEBUG Forward: Last Time Step Hidden State Shape {lastTimeStepHiddenState.shape}");

                // 7. Passar pela Camada Linear Final
                logits = linearOutput.forward(lastTimeStepHiddenState);
                 // Console.WriteLine($"DEBUG Forward: Final Logits Shape {logits.shape}");

                if ((bool)(logits == null)) {
                    throw new InvalidOperationException("Logits became null after linear layer.");
                }

                // 8. RETORNAR OS LOGITS
                return logits;

            }
            catch (Exception ex) { /* ... Bloco catch ... */ throw; }
            finally
            {
                // Dispose
                embedded?.Dispose();
                lstmInput?.Dispose();
                lstmOutputSequence?.Dispose();
                h_n?.Dispose(); // Descarta h_n
                lastTimeStepHiddenState?.Dispose();
            }
        } // Fim forward
    }
}