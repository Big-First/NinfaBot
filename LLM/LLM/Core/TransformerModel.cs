using Console = System.Console;
using Exception = System.Exception;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn; // Para ModuleList, Linear, Embedding, LayerNorm

namespace LLM.Core;

public class TransformerModel : Module
{
    private readonly Embedding tokenEmbedding;
    private readonly Embedding positionEmbedding; // Embedding posicional aprendível
    private readonly ModuleList<Module> transformerBlocks = new(); // Usar ModuleList genérico
    private readonly LayerNorm finalNorm; // Normalização final antes da saída
    private readonly Linear outputLinear; // Camada de saída para logits

    // Tornar maxSeqLen e vocabSize públicos para acesso pelo Trainer/Inferência
    public readonly int maxSeqLen;
    public readonly int embeddingDim;
    public readonly int vocabSize;
    public Device device;

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
        device = cuda.is_available() ? CUDA : CPU;
        this.vocabSize = vocabSize;
        this.maxSeqLen = maxSeqLen;
        this.embeddingDim = embeddingDim;

        tokenEmbedding = Embedding(vocabSize, embeddingDim);
        positionEmbedding = Embedding(maxSeqLen, embeddingDim);

        // Instanciar usando o tipo genérico torch.nn.Module
        transformerBlocks = new ModuleList<torch.nn.Module>();
        for (int i = 0; i < numLayers; i++)
        {
            transformerBlocks.Append(new TransformerBlock($"Block_{i}", embeddingDim, numHeads, maxSeqLen, dropout));
        }

        finalNorm = LayerNorm(embeddingDim);
        outputLinear = Linear(embeddingDim, vocabSize);

        // Inicialização de pesos (opcional, mas recomendado para estabilidade)
        // foreach (var p in this.parameters())
        // {
        //      if (p.IsInitialized) // Check if parameter requires gradients and is not frozen
        //      {
        //          // Exemplo de inicialização Kaiming Uniforme (comum para ReLU/GELU)
        //          torch.nn.init.kaiming_uniform_(p, nonlinearity: torch.nn.init.Nonlinearity.ReLU);
        //          // Para camadas de normalização ou bias, pode ser diferente
        //      }
        // }


        RegisterComponents(); // Garante que módulos filhos são registrados e movidos para o device
    }

    public Tensor forward(Tensor input)
    {
        var (batchSize, seqLen) = (input.shape[0], input.shape[1]);

        // O forward do modelo espera uma sequência de tamanho ATÉ maxSeqLen.
        // Se a sequência de entrada for maior, precisamos TRUNCAR aqui.
        // O endpoint de Inferência agora passa a sequência gerada COMPLETA,
        // então o truncamento deve ocorrer DENTRO do forward do modelo.
        if (seqLen > maxSeqLen)
        {
            // Truncar a sequência de entrada para pegar APENAS os últimos maxSeqLen tokens
            long start = seqLen - maxSeqLen; // Calcula o índice inicial
            long end = seqLen;               // O índice final (exclusivo)
            long step = 1;                   // O passo

            // Slicing em TorchSharp: tensor.slice(dim, start, end, step)
            input = input.slice(1, start, end, step);

            seqLen = maxSeqLen; // Atualiza seqLen para o novo comprimento
            // Console.WriteLine($"Aviso: Sequência de entrada truncada para {maxSeqLen} tokens.");
        }
        // Se a sequência for menor que maxSeqLen, o embedding posicional
        // ainda usará posições de 0 a seqLen-1. Não precisa de padding aqui,
        // pois a máscara causal já lida com o triângulo inferior.

        var device = input.device; // Obter o dispositivo do tensor de entrada

        // Criar tensor de posições dinamicamente NO MESMO DISPOSITIVO DO INPUT
        var positions = torch.arange(0, seqLen, dtype: ScalarType.Int64, device: device)
            .unsqueeze(0) // [1, seqLen]
            .expand(batchSize, seqLen); // [batch, seqLen]

        // Calcular embeddings + posições
        var tokenEmbeds = tokenEmbedding.forward(input); // [batch, seqLen, embedDim]
        var posEmbeds = positionEmbedding.forward(positions); // [batch, seqLen, embedDim]
        var x = tokenEmbeds + posEmbeds; // [batch, seqLen, embedDim]

        // Aplicar dropout no embedding (comum)
        // Use this.training para aplicar dropout apenas durante o treino
        // x = torch.nn.functional.dropout(x, p: dropout, training: this.training); // Use o 'dropout' configurado no construtor

        // Passar pelos blocos Transformer
        foreach (var block in transformerBlocks)
        {
            dynamic dynamicBlock = block;
            x = dynamicBlock.forward(x);
        }

        // Normalização final
        x = finalNorm.forward(x);

        // Camada linear final para obter logits sobre o vocabulário
        var logits = outputLinear.forward(x); // [batch, seqLen, vocabSize]

        return logits;
    }

    // Tornar as propriedades maxSeqLen e vocabSize acessíveis publicamente
     public int MaxSeqLen => maxSeqLen;
     public int VocabSize => vocabSize;


    public void Save(string path)
    {
        try
        {
            // CORREÇÃO: Gerenciar a memória dos tensores NO state_dict com DisposeScope
            // A declaração 'using' aplica-se ao DisposeScope, NÃO ao stateDict
            this.save(path);

            Console.WriteLine($"Modelo salvo (usando Module.save) em: {path}");

            // O scope.Dispose() será chamado automaticamente aqui ao sair do using block,
            // liberando os tensores que foram registrados quando state_dict() foi chamado.
        }
        catch (Exception ex)
        {
            Console.WriteLine($"ERRO ao salvar o modelo em {path}: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }

    public void Load(string path)
    {
        if (!File.Exists(path))
        {
             Console.WriteLine($"AVISO: Arquivo de modelo não encontrado em {path}. Não foi possível carregar.");
             return;
        }

        try
        {
            device = TorchSharp.torch.cuda.is_available() ? CUDA : CPU;
            this.to(device); // Mover modelo para o device ANTES de carregar

            // CORREÇÃO: Usar o método Load do próprio Module
            // Este método carrega o estado COMPLETO do módulo a partir de um arquivo salvo por Module.save
            // Verificar se na sua versão (0.105.0) o método load tem um parâmetro 'strict'.
            // Se o erro "Cannot resolve symbol 'strict'" ocorrer aqui, remova ', strict: false'.
            this.load(path, strict: false); // Tentar com strict: false primeiro

            Console.WriteLine($"Modelo carregado (usando Module.load) de {path}");
        }
        catch (Exception ex)
        {
             Console.WriteLine($"ERRO geral ao carregar modelo de {path}: {ex.Message}");
             Console.WriteLine(ex.StackTrace);
             // Considere lançar a exceção ou ter um comportamento de fallback
             // throw;
        }
    }
}
// --- END OF FILE TransformerModel.cs ---