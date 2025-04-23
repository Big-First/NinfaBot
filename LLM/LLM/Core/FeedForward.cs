using TorchSharp;
using static TorchSharp.torch;

namespace LLM.Core;

public class FeedForward : nn.Module
{
    private readonly nn.Sequential net;

    // Dropout pode ser adicionado como parâmetro
    public FeedForward(string name, int embedDim, double dropout = 0.1) : base(name)
    {
        net = nn.Sequential(
            ("linear1", nn.Linear(embedDim, 4 * embedDim)),
            ("activation", nn.GELU()), // GELU é comum em Transformers modernos
            ("linear2", nn.Linear(4 * embedDim, embedDim)),
            ("dropout", nn.Dropout(dropout)) // Dropout após a segunda linear é comum
        );
        RegisterComponents();
    }

    public override Tensor forward(Tensor x) => net.forward(x);
}
// --- END OF FILE FeedForward.cs ---