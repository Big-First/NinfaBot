using TorchSharp;
using static TorchSharp.torch;

namespace LLM.Core;

public class MultiHeadSelfAttention : nn.Module
{
    private readonly int embedDim, numHeads, headDim;
    private readonly nn.Linear qkv, proj;
    private readonly Tensor mask; // Máscara causal pré-calculada

    // Adicione maxSeqLen ao construtor se quiser que a máscara seja baseada nele
    public MultiHeadSelfAttention(string name, int embedDim, int numHeads, int maxSeqLen = 512) : base(name)
    {
        if (embedDim % numHeads != 0)
            throw new ArgumentException("embedDim deve ser divisível por numHeads");

        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.headDim = embedDim / numHeads;

        qkv = nn.Linear(embedDim, embedDim * 3);
        proj = nn.Linear(embedDim, embedDim);

        // Pré-calcula a máscara causal para eficiência.
        // Certifique-se de que maxSeqLen seja grande o suficiente para suas necessidades.
        mask = torch.tril(torch.ones(1, 1, maxSeqLen, maxSeqLen, dtype: torch.@bool)) == false; // Usar bool para máscara
                                                                                               // Onde mask é true, o valor será substituído
        RegisterBuffer("causal_mask", mask); // Registrar como buffer não treinável

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        int B = x.shape[0]; // Batch size
        int T = x.shape[1]; // Sequence length
        int D = embedDim;   // Embedding dimension

        var qkvOut = qkv.forward(x); // (B, T, 3*D)

        // Separa Q, K, V
        var q = qkvOut.slice(2, 0 * D, 1 * D); // (B, T, D)
        var k = qkvOut.slice(2, 1 * D, 2 * D); // (B, T, D)
        var v = qkvOut.slice(2, 2 * D, 3 * D); // (B, T, D)

        // Remodelar e transpor para Multi-Head: (B, H, T, Dh)
        q = q.view(B, T, numHeads, headDim).transpose(1, 2);
        k = k.view(B, T, numHeads, headDim).transpose(1, 2);
        v = v.view(B, T, numHeads, headDim).transpose(1, 2);

        // Calcular scores de atenção: Q * K^T / sqrt(Dh)
        // (B, H, T, Dh) @ (B, H, Dh, T) -> (B, H, T, T)
        var scores = torch.matmul(q, k.transpose(-2, -1)) / Math.Sqrt(headDim);

        // Aplicar máscara causal: impede atenção a tokens futuros
        // Pega a sub-máscara relevante para o tamanho T atual
        var causalMaskSlice = mask[":", ":", $":{T}", $":{T}"].to(device: scores.device); // (1, 1, T, T)
        scores = scores.masked_fill(causalMaskSlice, float.NegativeInfinity);

        // Aplicar softmax para obter pesos de atenção
        var attn = torch.nn.functional.softmax(scores, dim: -1);
        // Aplicar dropout na atenção se necessário (comum em Transformers)
        // attn = torch.nn.functional.dropout(attn, p: 0.1, training: this.training);

        // Calcular saída ponderada: Attention * V
        // (B, H, T, T) @ (B, H, T, Dh) -> (B, H, T, Dh)
        var outTensor = torch.matmul(attn, v);

        // Remodelar de volta para (B, T, D)
        outTensor = outTensor.transpose(1, 2).contiguous().view(B, T, D);

        // Projeção final
        return proj.forward(outTensor);
    }
}
// --- END OF FILE MultiHeadSelfAttention.cs ---