using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn; // Necessário para nn.Linear, nn.Dropout, nn.Module (para herança e métodos de fábrica)
using System; // Necessário para Math.Sqrt, float.NegativeInfinity, ArgumentException
using TorchSharp.Modules; // NECESSÁRIO para os TIPOS de classe: Linear, Dropout, LayerNorm, Embedding, etc.

namespace LLM.Core;

public class MultiHeadSelfAttention : Module // Herda de nn.Module
{
    private readonly int embedDim, numHeads, headDim;
    private readonly Linear qkv, proj; // Camadas Lineares (Tipo Linear do TorchSharp.Modules)
    private readonly Dropout dropout; // Camada de Dropout (Tipo Dropout do TorchSharp.Modules) // <-- ESTA É A VARIÁVEL PARA A INSTÂNCIA DO MÓDULO
    private readonly Tensor mask; // Máscara causal pré-calculada (Tipo Tensor do TorchSharp)
    private readonly double dropoutProb; // Probabilidade de dropout (um double) // <-- ESTA É A VARIÁVEL PARA O VALOR NUMÉRICO


    // Adicione maxSeqLen e dropout ao construtor
    public MultiHeadSelfAttention(string name, int embedDim, int numHeads, int maxSeqLen = 512, double dropout = 0.1) : base(name)
    {
        if (embedDim % numHeads != 0)
            throw new ArgumentException("embedDim deve ser divisível por numHeads");

        this.embedDim = embedDim;
        this.dropoutProb = dropout; // Salvar probabilidade de dropout NO CAMPO dropoutProb

        // Inicializar numHeads e headDim
        this.numHeads = numHeads;
        this.headDim = embedDim / numHeads;


        qkv = Linear(embedDim, embedDim * 3); // Projeta para Q, K, V (concatenados)
        proj = Linear(embedDim, embedDim);   // Projeção final da saída da atenção

        // CORREÇÃO AQUI: Atribuir o resultado de Dropout(dropoutProb) PARA A VARIÁVEL 'dropout'
        dropout = Dropout(dropoutProb); // Chama o método de fábrica nn.Dropout(), retorna um objeto Dropout

        // Pré-calcula a máscara causal (triângulo superior)
        mask = torch.tril(torch.ones(maxSeqLen, maxSeqLen, dtype: torch.@bool)).logical_not(); // Cria máscara booleana

        RegisterBuffer("causal_mask", mask);

        RegisterComponents(); // ESSENCIAL: Registra os sub-módulos e buffers
    }

    public override Tensor forward(Tensor x)
    {
        // x shape: (B, T, D)
        int B = x.shape[0];
        int T = x.shape[1];
        int D = embedDim;

        var current_mask = this.get_buffer("causal_mask").AsTensor().to(device: x.device);

        var qkvOut = qkv.forward(x);

        var q = qkvOut.slice(2, 0 * headDim * numHeads, 1 * headDim * numHeads).view(B, T, numHeads, headDim).transpose(1, 2);
        var k = qkvOut.slice(2, 1 * headDim * numHeads, 2 * headDim * numHeads).view(B, T, numHeads, headDim).transpose(1, 2);
        var v = qkvOut.slice(2, 2 * headDim * numHeads, 3 * headDim * numHeads).view(B, T, numHeads, headDim).transpose(1, 2);

        var scores = torch.matmul(q, k.transpose(-2, -1)) / Math.Sqrt(headDim);

        var causalMaskSlice = current_mask[$":{T}", $":{T}"].unsqueeze(0).unsqueeze(0);
        scores = scores.masked_fill(causalMaskSlice, float.NegativeInfinity);

        var attn_weights = torch.nn.functional.softmax(scores, dim: -1);

        // A chamada dropout.forward(attn_weights) está correta, usando a instância do módulo Dropout
        var attn_output_weighted = dropout.forward(attn_weights);


        var outTensor = torch.matmul(attn_output_weighted, v);
        outTensor = outTensor.transpose(1, 2).contiguous().view(B, T, D);

        return proj.forward(outTensor);
    }
}
// --- END OF FILE MultiHeadSelfAttention.cs ---