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
    private readonly ModuleList<Module> transformerBlocks = new();
    private readonly LayerNorm finalNorm; // Normalização final antes da saída
    private readonly Linear outputLinear; // Camada de saída para logits

    private readonly int maxSeqLen;
    private readonly int embeddingDim;
    private readonly int vocabSize;

    public TransformerModel(
        string name,
        int vocabSize,
        int maxSeqLen = 512,
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
        positionEmbedding = Embedding(maxSeqLen, embeddingDim);

        // CORREÇÃO AQUI: Instanciar usando o tipo genérico
        transformerBlocks = new ModuleList<torch.nn.Module>();
        for (int i = 0; i < numLayers; i++)
        {
            transformerBlocks.Append(new TransformerBlock($"Block_{i}", embeddingDim, numHeads, maxSeqLen, dropout));
        }

        finalNorm = LayerNorm(embeddingDim);
        outputLinear = Linear(embeddingDim, vocabSize);

        RegisterComponents();
    }

    public Tensor forward(Tensor input)
    {
        var (batchSize, seqLen) = (input.shape[0], input.shape[1]);

        if (seqLen > maxSeqLen)
        {
            // Truncar ou lançar erro se a sequência for muito longa
            long start = seqLen - maxSeqLen; // Calcula o índice inicial para pegar os últimos maxSeqLen tokens
            long end = seqLen;               // O índice final (exclusivo) é o comprimento original
            long step = 1;                   // O passo é 1 para pegar tokens consecutivos

            // CORREÇÃO AQUI: Adiciona o argumento step=1
            input = input.slice(1, start, end, step);

            seqLen = maxSeqLen; // Atualiza seqLen para o novo comprimento
            //Console.WriteLine($"Aviso: Sequência de entrada truncada para {maxSeqLen} tokens.");
        }

        var device = input.device;
        // Obter o dispositivo do tensor de entrada

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