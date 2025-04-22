// Core/TransformerModel.cs - Com Dropout para Regularização

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
        private readonly PositionalEncoding posEncoder; // Já contém dropout
        private readonly ModuleList<TransformerDecoderLayer> decoderLayers;
        private readonly LayerNorm norm;
        private readonly Linear outputLayer;
        private readonly int paddingIdx;
        public Device device = torch.CPU;

        // Construtor recebe dropoutRate
        public TransformerModel(int vocabSize, int dModel, int nhead, int numDecoderLayers, int dimFeedforward, double dropoutRate, int paddingIdx)
            : base(nameof(TransformerModel))
        {
            this.dModel = dModel;
            this.paddingIdx = paddingIdx;

            this.tokenEmbedding = Embedding(vocabSize, dModel, padding_idx: this.paddingIdx);
            // Passa dropoutRate para PositionalEncoding (ele aplica internamente)
            this.posEncoder = new PositionalEncoding(dModel, dropoutRate);
            this.decoderLayers = ModuleList<TransformerDecoderLayer>();
            for (int i = 0; i < numDecoderLayers; i++)
            {
                // Passa dropoutRate para cada TransformerDecoderLayer
                var decoderLayerInstance = TransformerDecoderLayer(
                    d_model: dModel,
                    nhead: nhead,
                    dim_feedforward: dimFeedforward,
                    dropout: dropoutRate // <<< Dropout da camada Transformer
                );
                this.decoderLayers.Add(decoderLayerInstance);
            }
            this.norm = LayerNorm(dModel);
            this.outputLayer = Linear(dModel, vocabSize);

            RegisterComponents(); // Garante que todas as camadas (incluindo dropout interno) sejam registradas
            Console.WriteLine($"TransformerModel Initialized with Dropout={dropoutRate}...");
        }

        public void SetDevice(Device targetDevice)
        {
            this.device = targetDevice ?? torch.CPU;
            Console.WriteLine($"TransformerModel internal device explicitly set to: {this.device}");
        }

        // ... (métodos de máscara como antes) ...
        private Tensor generate_square_subsequent_mask(long sz, Device device)
        {
            var mask = torch.triu(torch.full(new long[] { sz, sz }, float.NegativeInfinity, device: device), diagonal: 1);
            return mask;
        }

        private Tensor generate_padding_mask(Tensor src_transposed) // Recebe (SeqLen, Batch)
        {
            // Retorna (Batch, SeqLen) - True onde for padding
            Tensor src_pad_mask = src_transposed.eq(this.paddingIdx).transpose(0, 1);
            return src_pad_mask;
        }

        public override Tensor forward(Tensor src)
        {
            // ... (lógica inicial de device/type/transpose como antes) ...
             Device currentDevice = this.device;
             ScalarType longType = ScalarType.Int64;
             Tensor srcProcessed = src.alias();
             // ... (código para garantir tipo e device em srcProcessed) ...
             Tensor src_transposed = srcProcessed.transpose(0, 1);
             long seqLen = src_transposed.shape[0];
             if (srcProcessed.Handle != src.Handle && srcProcessed.Handle != src_transposed.Handle) srcProcessed.Dispose();
             else if (srcProcessed.Handle != src_transposed.Handle) srcProcessed.Dispose(); // Dispose alias

            Tensor? tgt_mask = null, src_key_padding_mask = null, embedded_src = null, src_with_pos = null;
            Tensor? decoderOutput = null, outputLogits = null;

            try
            {
                tgt_mask = generate_square_subsequent_mask(seqLen, currentDevice);
                src_key_padding_mask = generate_padding_mask(src_transposed);

                // Embedding + Positional Encoding (posEncoder já aplica dropout)
                embedded_src = this.tokenEmbedding.forward(src_transposed) * Math.Sqrt(this.dModel);
                src_with_pos = this.posEncoder.forward(embedded_src);

                // Camadas Decoder (TransformerDecoderLayer já aplica dropout interno)
                decoderOutput = src_with_pos.alias();
                foreach (var layer in this.decoderLayers) {
                     var currentLayerInput = decoderOutput.alias();
                     if (decoderOutput.Handle != src_with_pos.Handle && decoderOutput.Handle != currentLayerInput.Handle) decoderOutput.Dispose();
                     decoderOutput = layer.forward(tgt: currentLayerInput, memory: src_with_pos, tgt_mask: tgt_mask, memory_mask: null, tgt_key_padding_mask: src_key_padding_mask, memory_key_padding_mask: src_key_padding_mask);
                     currentLayerInput.Dispose();
                 }

                // Normalização Final
                if (this.norm != null) {
                     var normedOutput = this.norm.forward(decoderOutput);
                     if (decoderOutput.Handle != normedOutput.Handle) decoderOutput.Dispose();
                     decoderOutput = normedOutput;
                }

                // Camada Linear de Saída
                outputLogits = this.outputLayer.forward(decoderOutput);

                // Retorna TODOS os logits (para o Trainer)
                return outputLogits;
            }
            finally { /* ... Dispose ... */ }
        }
    } // --- Fim Classe ---
} // --- Fim Namespace ---

// --- Lembre-se de ajustar PositionalEncoding.cs também se necessário ---
// Core/PositionalEncoding.cs (Revisão - Dropout já estava lá)
