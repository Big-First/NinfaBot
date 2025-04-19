using System;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ChatBotAPI.Core
{
    public class TransformerModel : Module<Tensor, Tensor>
    {
        private readonly int dModel;
        private readonly Embedding tokenEmbedding;
        private readonly PositionalEncoding posEncoder;
        private readonly ModuleList<TransformerDecoderLayer> decoderLayers;
        private readonly LayerNorm norm;
        private readonly Linear outputLayer;
        private readonly int paddingIdx;
        public Device device = torch.CPU; // Default

        public TransformerModel(int vocabSize, int dModel, int nhead, int numDecoderLayers, int dimFeedforward, double dropoutRate, int paddingIdx)
            : base(nameof(TransformerModel))
        {
            this.dModel = dModel;
            this.paddingIdx = paddingIdx;

            this.tokenEmbedding = Embedding(vocabSize, dModel, padding_idx: this.paddingIdx);
            this.posEncoder = new PositionalEncoding(dModel, dropoutRate); // Usa a versão corrigida/aprendível
            this.decoderLayers = ModuleList<TransformerDecoderLayer>();
            for (int i = 0; i < numDecoderLayers; i++)
            {
                // Usa construtor mínimo que funciona na sua versão do TorchSharp
                var decoderLayerInstance = TransformerDecoderLayer(
                    d_model: dModel, nhead: nhead, dim_feedforward: dimFeedforward, dropout: dropoutRate
                );
                this.decoderLayers.Add(decoderLayerInstance);
            }
            this.norm = LayerNorm(dModel);
            this.outputLayer = Linear(dModel, vocabSize);
            RegisterComponents();
            Console.WriteLine($"TransformerModel Initialized: Vocab={vocabSize}, DModel={dModel}, Heads={nhead}, Layers={numDecoderLayers}, FFDim={dimFeedforward}, Dropout={dropoutRate}, PadIdx={paddingIdx}, BatchFirst=False");
        }

        public void SetDevice(Device targetDevice)
        {
            this.device = targetDevice ?? torch.CPU;
            Console.WriteLine($"TransformerModel internal device explicitly set to: {this.device}");
        }

        // Gera a máscara causal NO DEVICE DO INPUT
        private Tensor generate_square_subsequent_mask(long sz, Device device)
        {
            var mask = torch.triu(torch.full(new long[] { sz, sz }, float.NegativeInfinity, device: device), diagonal: 1);
            return mask;
        }

        // Gera a máscara de padding NO DEVICE DO INPUT
        private Tensor generate_padding_mask(Tensor src_transposed) // Recebe src já transposto (SeqLen, Batch)
        {
             Tensor src_pad_mask = src_transposed.eq(this.paddingIdx).transpose(0, 1); // Output (Batch, SeqLen)
             return src_pad_mask;
        }


        // --- Método forward AJUSTADO para retornar TODOS os logits ---
        public override Tensor forward(Tensor src) // src chega com shape (Batch, SeqLen)
        {
            Device currentDevice = this.device; // Usa o device configurado para o módulo
            ScalarType longType = ScalarType.Int64;

            // Garante tipo e device (fazendo em 2 passos)
            Tensor srcProcessed = src.alias();
            bool typeConverted = false;
            bool movedDevice = src.device.type != currentDevice.type || src.device.index != currentDevice.index;

            if (src.dtype != longType) {
                var temp = srcProcessed.alias();
                srcProcessed.Dispose();
                srcProcessed = temp.to(longType);
                typeConverted = true;
                temp.Dispose();
                if(movedDevice) Console.WriteLine("Warning: Input converted to Int64."); // Log apenas se ambos mudaram
            }
            if (movedDevice) {
                var temp = srcProcessed.alias();
                srcProcessed.Dispose();
                srcProcessed = temp.to(currentDevice);
                temp.Dispose();
                 Console.WriteLine($"Warning: Input tensor moved to device {currentDevice}.");
            }

            // Transpõe para (SeqLen, BatchSize)
            Tensor src_transposed = srcProcessed.transpose(0, 1);
            long seqLen = src_transposed.shape[0];

            // Dispose do tensor processado se ele for diferente do input original
             if (srcProcessed.Handle != src.Handle) srcProcessed.Dispose();

            Tensor? tgt_mask = null, src_key_padding_mask = null, embedded_src = null, src_with_pos = null;
            Tensor? decoderOutput = null, outputLogits = null; // Não precisamos mais de lastTokenLogits aqui

            try
            {
                // 1. Gerar Máscaras
                tgt_mask = generate_square_subsequent_mask(seqLen, currentDevice);
                src_key_padding_mask = generate_padding_mask(src_transposed);

                // 2. Embedding + Positional Encoding
                embedded_src = this.tokenEmbedding.forward(src_transposed) * Math.Sqrt(this.dModel);
                src_with_pos = this.posEncoder.forward(embedded_src);

                // 3. Passar pelas Camadas Decoder
                decoderOutput = src_with_pos.alias();
                foreach (var layer in this.decoderLayers) {
                    var currentLayerInput = decoderOutput.alias();
                    decoderOutput.Dispose();
                    decoderOutput = layer.forward(tgt: currentLayerInput, memory: src_with_pos, tgt_mask: tgt_mask, memory_mask: null, tgt_key_padding_mask: src_key_padding_mask, memory_key_padding_mask: src_key_padding_mask);
                    currentLayerInput.Dispose();
                 }

                // 4. Normalização Final
                if (this.norm != null) {
                     var normedOutput = this.norm.forward(decoderOutput);
                     decoderOutput.Dispose();
                     decoderOutput = normedOutput;
                }

                // 5. Camada Linear de Saída
                outputLogits = this.outputLayer.forward(decoderOutput); // Shape (SeqLen, Batch, VocabSize)

                // *** CORREÇÃO: Retorna TODOS os logits da sequência ***
                return outputLogits;
            }
            finally
            {
                 // Dispose
                 src_transposed?.Dispose();
                 tgt_mask?.Dispose();
                 src_key_padding_mask?.Dispose();
                 embedded_src?.Dispose();
                 src_with_pos?.Dispose();
                 decoderOutput?.Dispose();
                 // Não descartamos outputLogits aqui, pois ele é o valor de retorno
                 // O chamador (Trainer) será responsável por descartá-lo.
            }
        } // --- Fim Forward ---

    } // --- Fim Classe ---
} // --- Fim Namespace ---