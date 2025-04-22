// --- START OF FILE TransformerModel.cs ---
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn; // Para ModuleList, Linear, Embedding, LayerNorm

namespace LLM.Core;

public class TransformerModel : Module
{
    private readonly Embedding tokenEmbedding;
    private readonly Embedding positionEmbedding; // Embedding posicional aprendível
    private readonly ModuleList transformerBlocks;
    private readonly LayerNorm finalNorm; // Normalização final antes da saída
    private readonly Linear outputLinear; // Camada de saída para logits

    private readonly int maxSeqLen;
    private readonly int embeddingDim;
    private readonly int vocabSize;

    public TransformerModel(
        string name,
        int vocabSize,
        int maxSeqLen = 512, // Aumentado o padrão
        int embeddingDim = 256,
        int numHeads = 4,
        int numLayers = 2,
        double dropout = 0.1)
        : base(name)
    {
        this.vocabSize = vocabSize;
        this.maxSeqLen = maxSeqLen;
        this.embeddingDim = embeddingDim;

        tokenEmbedding = Embedding(vocabSize, embeddingDim);
        // Embedding posicional aprendível é comum e funciona bem
        positionEmbedding = Embedding(maxSeqLen, embeddingDim);

        transformerBlocks = new ModuleList();
        for (int i = 0; i < numLayers; i++)
        {
            // Passar maxSeqLen para o bloco, que passará para a atenção
            transformerBlocks.Append(new TransformerBlock($"Block_{i}", embeddingDim, numHeads, maxSeqLen, dropout));
        }

        // Adicionar LayerNorm final é uma prática comum
        finalNorm = LayerNorm(embeddingDim);
        outputLinear = Linear(embeddingDim, vocabSize);

        // Inicialização de pesos (opcional, mas pode ajudar)
        // this.apply(InitWeights);

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var (batchSize, seqLen) = (input.shape[0], input.shape[1]);

        if (seqLen > maxSeqLen)
        {
             // Truncar ou lançar erro se a sequência for muito longa
             input = input.slice(1, seqLen - maxSeqLen, seqLen);
             seqLen = maxSeqLen;
             //Console.WriteLine($"Aviso: Sequência de entrada truncada para {maxSeqLen} tokens.");
        }


        var device = input.device; // Obter o dispositivo do tensor de entrada

        // Criar tensor de posições dinamicamente
        var positions = torch.arange(0, seqLen, dtype: ScalarType.Int64, device: device)
            .unsqueeze(0) // [1, seqLen]
            .expand(batchSize, seqLen); // [batch, seqLen]

        // Calcular embeddings + posições
        var tokenEmbeds = tokenEmbedding.forward(input); // [batch, seqLen, embedDim]
        var posEmbeds = positionEmbedding.forward(positions); // [batch, seqLen, embedDim]
        var x = tokenEmbeds + posEmbeds; // [batch, seqLen, embedDim]

        // Aplicar dropout no embedding (comum)
        // x = torch.nn.functional.dropout(x, p: 0.1, training: this.training);

        // Passar pelos blocos Transformer
        foreach (var block in transformerBlocks)
        {
            x = block.forward(x);
        }

        // Normalização final
        x = finalNorm.forward(x);

        // Camada linear final para obter logits sobre o vocabulário
        var logits = outputLinear.forward(x); // [batch, seqLen, vocabSize]

        return logits;
    }

    // Opcional: Função para inicializar pesos
    // private void InitWeights(Module module)
    // {
    //     if (module is Linear linear)
    //     {
    //         torch.nn.init.normal_(linear.weight, mean: 0.0, std: 0.02);
    //         if (linear.bias is not null)
    //         {
    //             torch.nn.init.zeros_(linear.bias);
    //         }
    //     }
    //     else if (module is Embedding embedding)
    //     {
    //         torch.nn.init.normal_(embedding.weight, mean: 0.0, std: 0.02);
    //     }
    // }

    public void Save(string path) => this.save(path);
    public void Load(string path)
    {
        try
        {
             this.load(path, skipIncompatibleKeys: true); // Permite carregar mesmo se houver pequenas diferenças
             Console.WriteLine($"Modelo carregado de {path}");
        }
        catch (Exception ex)
        {
             Console.WriteLine($"Erro ao carregar modelo de {path}: {ex.Message}. Verifique o caminho e a compatibilidade do modelo.");
             // Considere lançar a exceção ou ter um comportamento de fallback
             // throw;
        }
    }
}
// --- END OF FILE TransformerModel.cs ---