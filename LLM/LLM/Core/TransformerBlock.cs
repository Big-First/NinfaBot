// --- START OF FILE TransformerBlock.cs ---
using TorchSharp;
using static TorchSharp.torch;

namespace LLM.Core;

public class TransformerBlock : nn.Module
{
    private readonly nn.LayerNorm norm1, norm2;
    private readonly MultiHeadSelfAttention attn;
    private readonly FeedForward ff;
    private readonly int embedDim;

    // Passar maxSeqLen para a atenção
    public TransformerBlock(string name, int embedDim, int numHeads, int maxSeqLen, double dropout = 0.1) : base(name)
    {
        this.embedDim = embedDim;
        norm1 = nn.LayerNorm(embedDim);
        norm2 = nn.LayerNorm(embedDim);
        // Passar maxSeqLen aqui
        attn = new MultiHeadSelfAttention($"{name}_attn", embedDim, numHeads, maxSeqLen);
        ff = new FeedForward($"{name}_ff", embedDim, dropout);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        // Atenção + Conexão Residual + Normalização
        var attnOutput = attn.forward(norm1.forward(x));
        x = x + attnOutput; // Conexão Residual 1

        // FeedForward + Conexão Residual + Normalização
        var ffOutput = ff.forward(norm2.forward(x));
        x = x + ffOutput; // Conexão Residual 2

        return x;
    }
}
// --- END OF FILE TransformerBlock.cs ---