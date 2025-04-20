// Core/TransformerModel.cs - AJUSTADO para retornar logits completos E corrigir .Handle

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
        private Device device = torch.CPU; // Default

        public TransformerModel(int vocabSize, int dModel, int nhead, int numDecoderLayers, int dimFeedforward, double dropoutRate, int paddingIdx)
            : base(nameof(TransformerModel))
        {
            this.dModel = dModel;
            this.paddingIdx = paddingIdx;

            this.tokenEmbedding = Embedding(vocabSize, dModel, padding_idx: this.paddingIdx);
            this.posEncoder = new PositionalEncoding(dModel, dropoutRate);
            this.decoderLayers = ModuleList<TransformerDecoderLayer>();
            for (int i = 0; i < numDecoderLayers; i++)
            {
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

        private Tensor generate_square_subsequent_mask(long sz, Device device)
        {
            var mask = torch.triu(torch.full(new long[] { sz, sz }, float.NegativeInfinity, device: device), diagonal: 1);
            return mask;
        }

        private Tensor generate_padding_mask(Tensor src_transposed)
        {
             Tensor src_pad_mask = src_transposed.eq(this.paddingIdx).transpose(0, 1);
             return src_pad_mask;
        }

        public override Tensor forward(Tensor src) // src chega com shape (Batch, SeqLen)
        {
            Device currentDevice = this.device;
            ScalarType longType = ScalarType.Int64;

            Tensor srcProcessed = src.alias();
            bool typeConverted = false;
            bool movedDevice = src.device.type != currentDevice.type || src.device.index != currentDevice.index;
            bool aliasUsed = true; // Assume que começamos com alias

            if (src.dtype != longType) {
                var temp = srcProcessed.alias(); // Cria alias do tensor atual (que pode ser o src original ou o movido)
                if (aliasUsed) srcProcessed.Dispose(); // Descarta alias anterior se houver
                srcProcessed = temp.to(longType); // Converte
                typeConverted = true;
                aliasUsed = false; // Agora é uma cópia ou novo tensor
                temp.Dispose(); // Descarta alias temporário
                if(movedDevice) Console.WriteLine("Warning: Input converted to Int64.");
            }
            if (movedDevice) {
                var temp = srcProcessed.alias(); // Cria alias do tensor atual
                if (!aliasUsed && srcProcessed.Handle != src.Handle) srcProcessed.Dispose(); // Descarta anterior se não era alias
                srcProcessed = temp.to(currentDevice); // Move
                aliasUsed = false; // Agora é uma cópia ou novo tensor
                temp.Dispose();
                 Console.WriteLine($"Warning: Input tensor moved to device {currentDevice}.");
            }

            // Transpõe para (SeqLen, BatchSize)
            Tensor src_transposed = srcProcessed.transpose(0, 1);
            long seqLen = src_transposed.shape[0];

            // *** CORREÇÃO: Usa 'handle' (minúsculo) para comparar ponteiros de tensores ***
            if (!aliasUsed && srcProcessed.Handle != src.Handle) {
                 Console.WriteLine("DEBUG: Disposing intermediate srcProcessed tensor."); // Log opcional
                 srcProcessed.Dispose(); // Descarta o tensor intermediário final se foi criado
            } else if (aliasUsed && srcProcessed.Handle != src_transposed.Handle) {
                 // Se era só alias E não é o mesmo handle que o transposto (raro), descarta o alias
                 srcProcessed.Dispose();
            }


            Tensor? tgt_mask = null, src_key_padding_mask = null, embedded_src = null, src_with_pos = null;
            Tensor? decoderOutput = null, outputLogits = null;

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
                    // Se decoderOutput não for o alias original de src_with_pos, descarte-o
                    if (decoderOutput.Handle != src_with_pos.Handle && decoderOutput.Handle != currentLayerInput.Handle) decoderOutput.Dispose();
                    decoderOutput = layer.forward(tgt: currentLayerInput, memory: src_with_pos, tgt_mask: tgt_mask, memory_mask: null, tgt_key_padding_mask: src_key_padding_mask, memory_key_padding_mask: src_key_padding_mask);
                    currentLayerInput.Dispose();
                 }

                // 4. Normalização Final
                if (this.norm != null) {
                     var normedOutput = this.norm.forward(decoderOutput);
                      // Descarta a saída anterior do loop decoder se ela for diferente da saída normalizada
                      if (decoderOutput.Handle != normedOutput.Handle) decoderOutput.Dispose();
                     decoderOutput = normedOutput;
                }

                // 5. Camada Linear de Saída
                outputLogits = this.outputLayer.forward(decoderOutput); // Shape (SeqLen, Batch, VocabSize)

                // *** Retorna TODOS os logits ***
                return outputLogits;
            }
            finally
            {
                 // Dispose dos tensores criados neste escopo
                 src_transposed?.Dispose();
                 tgt_mask?.Dispose();
                 src_key_padding_mask?.Dispose();
                 embedded_src?.Dispose();
                 src_with_pos?.Dispose(); // A memória original
                 decoderOutput?.Dispose(); // A saída final do decoder/norm antes da camada linear
                 // outputLogits é retornado, não descartado aqui
            }
        } // --- Fim Forward ---

    } // --- Fim Classe ---
} // --- Fim Namespace ---