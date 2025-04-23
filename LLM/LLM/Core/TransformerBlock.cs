using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace LLM.Core;

public class TransformerBlock : Module
{
    private readonly LayerNorm norm1, norm2;
    private readonly MultiHeadSelfAttention attn;
    private readonly FeedForward ff;
    private readonly int embedDim;
    private readonly double dropoutProb; // Adicionar probabilidade de dropout
    private readonly Dropout dropout1, dropout2; // Camadas de dropout para conexões residuais

    // Passar maxSeqLen e dropout para as subcamadas
    public TransformerBlock(string name, int embedDim, int numHeads, int maxSeqLen, double dropout = 0.1) : base(name)
    {
        this.embedDim = embedDim;
        this.dropoutProb = dropout; // Salvar probabilidade de dropout

        norm1 = LayerNorm(embedDim);
        attn = new MultiHeadSelfAttention($"{name}_attn", embedDim, numHeads, maxSeqLen, dropoutProb); // Passar dropout
        dropout1 = nn.Dropout(dropoutProb); // Dropout para a primeira conexão residual

        norm2 = LayerNorm(embedDim);
        ff = new FeedForward($"{name}_ff", embedDim, dropoutProb); // Passar dropout
        dropout2 = Dropout(dropoutProb); // Dropout para a segunda conexão residual

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        var residual1 = x;
        // Tentativa de cast explícito para diagnóstico
        var attnOutput = ((Module)attn).forward(((Module)norm1).forward(x));
        x = residual1 + ((Module)dropout1).forward(attnOutput);

        var residual2 = x;
        // Tentativa de cast explícito para diagnóstico
        var ffOutput = ((Module)ff).forward(((Module)norm2).forward(x));
        x = residual2 + ((Module)dropout2).forward(ffOutput);

        return x;
    }
}
// --- END OF FILE TransformerBlock.cs ---