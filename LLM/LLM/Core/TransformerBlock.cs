using TorchSharp;
using TorchSharp.Modules;
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

    public Tensor forward(Tensor x)
    {
        var residual1 = x;
        var norm1_out = norm1.forward(x); // norm1 é LayerNorm, herda de Module. Forward deve ser resolvido.
        var attnOutput = attn.forward(norm1_out); // attn é MultiHeadSelfAttention, DEVE herdar de Module. Forward deve ser resolvido.
        var dropout1_out = dropout1.forward(attnOutput); // dropout1 é Dropout, herda de Module. Forward deve ser resolvido.
        x = residual1 + dropout1_out;

        var residual2 = x;
        var norm2_out = norm2.forward(x); // norm2 é LayerNorm
        var ffOutput = ff.forward(norm2_out); // ff é FeedForward, DEVE herdar de Module
        var dropout2_out = dropout2.forward(ffOutput); // dropout2 é Dropout
        x = residual2 + dropout2_out;

        return x;
    }
}
// --- END OF FILE TransformerBlock.cs ---