// TorchSharpModel.cs - VERSÃO LSTM CORRIGIDA

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
            this.hiddenSize = hiddenSize;
            this.paddingIdx = paddingIdx;
            this.numLSTMLayers = numLSTMLayers;
            this.embedding = Embedding(vocabSize, embeddingSize, padding_idx: this.paddingIdx);
            // batchFirst = false significa que o formato esperado é (SeqLen, BatchSize, InputSize)
            this.lstm = LSTM(inputSize: embeddingSize, hiddenSize: this.hiddenSize, numLayers: this.numLSTMLayers, batchFirst: false);
            this.linearOutput = Linear(inputSize: this.hiddenSize, outputSize: vocabSize);
            RegisterComponents();
            Console.WriteLine($"TorchSharpModel (LSTM) Initialized: VocabSize={vocabSize}, EmbeddingSize={embeddingSize}, HiddenSize={this.hiddenSize}, LSTMLayers={this.numLSTMLayers}, PaddingIdx={this.paddingIdx}");
        }

        // --- Método forward com LSTM (CORRIGIDO) ---
        // Espera input de shape (SeqLen, BatchSize). Em inferência BatchSize=1.
        public override Tensor forward(Tensor input)
        {
            // DECLARAÇÃO FORA DO TRY PARA ACESSO NO FINALLY
            Tensor? embedded = null;
            // Declarar como uma tupla de 3 Tensors (output_seq, h_n, c_n), anulável
            (Tensor output_seq, Tensor h_n, Tensor c_n)? lstmResult = null;
            Tensor? lstmOutputSequence = null; // Irá referenciar lstmResult.Value.output_seq
            Tensor? lastTimeStepHiddenState = null;
            Tensor? logits = null; // Será retornado no caminho de sucesso

            try
            {
                // 1. Garante tipo Long
                if (input.dtype != ScalarType.Int64) { input = input.to(ScalarType.Int64); }

                // Console.WriteLine($"DEBUG TorchSharpModel Forward: Input shape before embedding: {input.shape}"); // Log de debug

                // 2. Embedding
                // Input (SeqLen, BatchSize) -> Embedded (SeqLen, BatchSize, EmbeddingSize)
                embedded = embedding.forward(input);

                // Console.WriteLine($"DEBUG TorchSharpModel Forward: Embedded shape before LSTM: {embedded.shape}"); // Log de debug

                // 3. Passar pela LSTM
                // A entrada do LSTM já está no formato correto (SeqLen, BatchSize, InputSize)
                // CORREÇÃO ANTERIOR CONFIRMADA: Não precisa de unsqueeze(1) aqui
                lstmResult = lstm.forward(embedded); // Atribuição à variável declarada fora do try

                // Verifica se o resultado do LSTM é nulo (não deveria acontecer, mas como segurança)
                 if (lstmResult == null) {
                     throw new InvalidOperationException("LSTM forward returned null.");
                 }

                // 4. Extrai output sequence
                // CORRIGIDO: Acessar output_seq usando a propriedade nomeada ou Item1
                // Não precisa mais da verificação HasValue aqui porque já verificamos acima que lstmResult != null
                lstmOutputSequence = lstmResult.Value.output_seq; // OU lstmResult.Value.Item1

                // Console.WriteLine($"DEBUG TorchSharpModel Forward: LSTM output sequence shape: {lstmOutputSequence.shape}"); // Log de debug

                // 5. Obter a Saída Relevante (último passo de tempo da sequência)
                // lstmOutputSequence shape (SeqLen, BatchSize, HiddenSize)
                // select(0, -1) -> (BatchSize, HiddenSize) (último elemento na dimensão SeqLen)
                // squeeze(0) -> (HiddenSize) (Remove a dimensão BatchSize=1, assumindo BatchSize=1)
                lastTimeStepHiddenState = lstmOutputSequence.select(0, -1).squeeze(0);

                // Console.WriteLine($"DEBUG TorchSharpModel Forward: Last Time Step Hidden State Shape before Linear: {lastTimeStepHiddenState.shape}"); // Log de debug

                // 6. Passar pela Camada Linear Final
                // Input (HiddenSize) -> Output (VocabSize)
                logits = linearOutput.forward(lastTimeStepHiddenState);

                // Console.WriteLine($"DEBUG TorchSharpModel Forward: Final Logits Shape: {logits.shape}"); // Log de debug

                // Verifica se logits foi criado
                if ((bool)(logits == null)) {
                     throw new InvalidOperationException("Linear layer returned null logits.");
                }

                // 7. RETORNAR OS LOGITS
                return logits; // RETORNA O TENSOR LOGITS

            }
            catch (Exception ex) {
                Console.Error.WriteLine($"FATAL ERROR in TorchSharpModel.forward: {ex.ToString()}");
                 // Garante dispose de TUDO em caso de erro antes de relançar
                 embedded?.Dispose();
                 lstmOutputSequence?.Dispose(); // Dispor o tensor output_seq
                 lastTimeStepHiddenState?.Dispose();
                 logits?.Dispose(); // Dispose do logits, pois não será retornado

                 // CORRIGIDO: Dispor h_n e c_n da tupla se lstmResult foi criado (usando HasValue)
                 if (lstmResult.HasValue)
                 {
                     lstmResult.Value.h_n?.Dispose();
                     lstmResult.Value.c_n?.Dispose();
                 }

                throw; // Relança a exceção
            }
            finally
            {
                // Dispose dos tensores intermediários que foram alocados NO TRY
                // e que NÃO SÃO o tensor de retorno (logits).
                // O tensor 'input' é de responsabilidade do chamador.

                embedded?.Dispose(); // Dispor o resultado do embedding
                lstmOutputSequence?.Dispose(); // Dispor o tensor output_seq (também lstmResult.Value.output_seq)
                lastTimeStepHiddenState?.Dispose(); // Dispor o estado extraído

                // CORRIGIDO: Dispor h_n e c_n da tupla se lstmResult foi criado (usando HasValue)
                if (lstmResult.HasValue)
                {
                    lstmResult.Value.h_n?.Dispose();
                    lstmResult.Value.c_n?.Dispose();
                }

                // logits NÃO é disposto aqui porque é o valor de retorno no caminho de sucesso.
            }
        } // Fim forward
    }
}