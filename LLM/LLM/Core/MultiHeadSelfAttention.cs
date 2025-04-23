using TorchSharp;
using static TorchSharp.torch;

namespace LLM.Core;

public class MultiHeadSelfAttention : nn.Module
{
    private readonly int embedDim, numHeads, headDim;
    private readonly nn.Linear qkv, proj;
    private readonly Tensor mask; // Máscara causal pré-calculada
    private readonly double dropoutProb; // Adicionar probabilidade de dropout
    private readonly nn.Dropout dropout; // Adicionar camada de dropout


    // Adicione maxSeqLen ao construtor se quiser que a máscara seja baseada nele
    public MultiHeadSelfAttention(string name, int embedDim, int numHeads, int maxSeqLen = 512, double dropout = 0.1) : base(name)
    {
        if (embedDim % numHeads != 0)
            throw new ArgumentException("embedDim deve ser divisível por numHeads");

        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.headDim = embedDim / numHeads;
        this.dropoutProb = dropout; // Salvar probabilidade de dropout

        qkv = nn.Linear(embedDim, embedDim * 3);
        proj = nn.Linear(embedDim, embedDim);
        dropout = nn.Dropout(dropoutProb); // Instanciar camada de dropout

        // Pré-calcula a máscara causal para eficiência.
        // Certifique-se de que maxSeqLen seja grande o suficiente para suas necessidades.
        // A máscara onde é TRUE, o valor SERÁ SUBSTITUÍDO (geralmente por -inf)
        mask = torch.tril(torch.ones(maxSeqLen, maxSeqLen, dtype: torch.@bool)).logical_not(); // Máscara triangular inferior negada
                                                                                             // Onde mask é true (acima da diagonal), o valor será substituído por -infinity

        // Registrar como buffer não treinável. O "1, 1" na dimensão é opcional aqui,
        // o slicing no forward vai ajustar de qualquer forma. Mantendo 2D (maxSeqLen, maxSeqLen)
        // é mais limpo, o slicing no forward adiciona as dims de batch/heads.
        RegisterBuffer("causal_mask", mask);


        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        int B = x.shape[0]; // Batch size
        int T = x.shape[1]; // Sequence length
        int D = embedDim;   // Embedding dimension

        // Mover a máscara para o mesmo dispositivo que o tensor de entrada 'x'
        var current_mask = this.get_buffer("causal_mask").AsTensor().to(device: x.device);

        var qkvOut = qkv.forward(x); // (B, T, 3*D)

        // Separa Q, K, V
        // O slicing precisa considerar o tamanho da dimensão 2, que é 3*D
        var q = qkvOut.slice(2, 0 * D, 1 * D).view(B, T, numHeads, headDim).transpose(1, 2); // (B, H, T, Dh)
        var k = qkvOut.slice(2, 1 * D, 2 * D).view(B, T, numHeads, headDim).transpose(1, 2); // (B, H, T, Dh)
        var v = qkvOut.slice(2, 2 * D, 3 * D).view(B, T, numHeads, headDim).transpose(1, 2); // (B, H, T, Dh)


        // Calcular scores de atenção: Q * K^T / sqrt(Dh)
        // (B, H, T, Dh) @ (B, H, Dh, T) -> (B, H, T, T)
        var scores = torch.matmul(q, k.transpose(-2, -1)) / Math.Sqrt(headDim);

        // Aplicar máscara causal: impede atenção a tokens futuros
        // Pega a sub-máscara relevante para o tamanho T atual, e adiciona dims para batch e heads
        // current_mask tem shape (maxSeqLen, maxSeqLen). Fatiamos para (T, T)
        // Depois adicionamos unsqueeze para obter (1, 1, T, T) para broadcasting
        var causalMaskSlice = current_mask[$":{T}", $":{T}"].unsqueeze(0).unsqueeze(0); // (1, 1, T, T)
        scores = scores.masked_fill(causalMaskSlice, float.NegativeInfinity);

        // Aplicar softmax para obter pesos de atenção
        var attn = torch.nn.functional.softmax(scores, dim: -1);
        // Aplicar dropout na atenção (comum em Transformers)
        // Usar this.training para aplicar dropout apenas durante o treino
        attn = dropout.forward(attn);

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