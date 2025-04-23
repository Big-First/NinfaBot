// --- START OF FILE Trainer.cs ---
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.nn;
using System; // Para Guid
using System.Collections.Generic; // Para List
using System.Linq; // Para Linq

namespace LLM.Core;

public class Trainer
{
    private readonly TransformerModel model;
    private readonly Tokenizer tokenizer;
    private readonly TrainerOptions options;
    private readonly Device device;
    private readonly Module lossFn;
    private readonly Optimizer optimizer;
    private readonly int padTokenId;

    public Trainer(TransformerModel model, Tokenizer tokenizer, TrainerOptions options)
    {
        this.model = model;
        this.tokenizer = tokenizer;
        this.options = options;
        this.device = torch.cuda.is_available() ? CUDA : CPU;
        this.padTokenId = tokenizer.GetPadTokenId(); // Obter ID de padding do tokenizer

        // Ignorar o token de padding no cálculo da loss
        this.lossFn = CrossEntropyLoss(ignore_index: padTokenId);
        this.optimizer = torch.optim.Adam(model.parameters(), lr: options.LearningRate);

        this.model.to(device); // Mover modelo para o dispositivo correto
        Console.WriteLine($"Treinamento será executado em: {device.type}");
    }

    public void Train(List<(string input, string output)> dataset)
    {
        if (dataset == null || dataset.Count == 0)
        {
            Console.WriteLine("Dataset de treinamento está vazio.");
            return;
        }

        Console.WriteLine($"🔧 Iniciando treinamento com {dataset.Count} exemplos...");
        Console.WriteLine($"   Épocas: {options.Epochs}, Tamanho do Lote: {options.BatchSize}, Seq Len: {options.MaxSeqLen}, LR: {options.LearningRate}");

        model.train(); // Colocar modelo em modo de treinamento

        for (int epoch = 1; epoch <= options.Epochs; epoch++)
        {
            double totalLoss = 0;
            int batchCount = 0;

            // Embaralhar o dataset a cada época
            var shuffledDataset = dataset.OrderBy(_ => Guid.NewGuid()).ToList();

            // Processar em lotes (batches)
            for (int i = 0; i < shuffledDataset.Count; i += options.BatchSize)
            {
                var batchItems = shuffledDataset.Skip(i).Take(options.BatchSize).ToList();
                if (!batchItems.Any()) continue;

                var batchInputs = new List<Tensor>();
                var batchTargets = new List<Tensor>();

                foreach (var (inputStr, outputStr) in batchItems)
                {
                    // Tokenizar Input e Target
                    var inputTokens = tokenizer.Encode(inputStr, addSpecialTokens: false); // Não adicionar BOS/EOS aqui
                    var targetTokens = tokenizer.Encode(outputStr, addSpecialTokens: false);
                    targetTokens.Add(tokenizer.GetEosTokenId()); // Adicionar EOS ao final do target

                    // Truncar se necessário (considerando espaço para EOS no target)
                    inputTokens = inputTokens.Take(options.MaxSeqLen).ToList();
                    // Target é deslocado, então tem o mesmo comprimento que input após adicionar EOS
                    targetTokens = targetTokens.Take(options.MaxSeqLen).ToList();

                    batchInputs.Add(torch.tensor(inputTokens, dtype: ScalarType.Int64));
                    batchTargets.Add(torch.tensor(targetTokens, dtype: ScalarType.Int64));
                }

                // Fazer Padding e criar tensores de lote
                var paddedInputs = PadBatch(batchInputs, options.MaxSeqLen, padTokenId).to(device);
                var paddedTargets = PadBatch(batchTargets, options.MaxSeqLen, padTokenId).to(device);

                // Etapa de Treinamento
                using var scope = torch.NewDisposeScope(); // Gerenciar memória do TorchSharp

                optimizer.zero_grad(); // Limpar gradientes anteriores

                var logits = model.forward(paddedInputs); // Forward pass [B, S, V]

                // Calcular Loss
                // Logits precisam ser [B*S, V]
                // Targets precisam ser [B*S]
                var loss = lossFn.forward(logits.view(-1, model.vocabSize), paddedTargets.view(-1));

                loss.backward(); // Calcular gradientes
                optimizer.step(); // Atualizar pesos

                totalLoss += loss.item<double>();
                batchCount++;

                // Limpar tensores do lote explicitamente (ajuda GC e memória GPU)
                paddedInputs.Dispose();
                paddedTargets.Dispose();
                logits.Dispose();
                loss.Dispose();
            } // Fim do loop de lotes

            double avgLoss = totalLoss / batchCount;
            Console.WriteLine($"📚 Época {epoch}/{options.Epochs} - Loss Média: {avgLoss:F4}");

        } // Fim do loop de épocas

        Console.WriteLine("✅ Treinamento finalizado.");
        try
        {
            model.Save(options.SavePath);
            Console.WriteLine($"💾 Modelo salvo em: {options.SavePath}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Erro ao salvar o modelo: {ex.Message}");
        }
    }

    /// <summary>
    /// Adiciona padding a uma lista de tensores para formar um lote.
    /// </summary>
    private Tensor PadBatch(List<Tensor> batch, int seqLen, int padValue)
    {
        var paddedTensors = batch.Select(t =>
        {
            var currentLen = t.shape[0];
            if (currentLen < seqLen)
            {
                // Cria um tensor de padding com o valor correto
                var paddingTensor = torch.full(new long[] { seqLen - currentLen }, padValue, dtype: t.dtype);
                // Concatena o tensor original com o padding
                var padded = torch.cat(new[] { t, paddingTensor }, dim: 0);
                paddingTensor.Dispose(); // Libera o tensor de padding intermediário
                return padded;
            }
            else if (currentLen > seqLen)
            {
                // Trunca o tensor se for maior que seqLen
                return t.slice(0, 0, seqLen);
            }
            else
            {
                // Nenhum padding/truncamento necessário, mas retorna uma cópia para consistência (?)
                // Ou apenas retorna o tensor original. Retornar original é mais eficiente.
                 return t;
                 // return t.clone(); // Se precisar garantir cópias separadas
            }
        }).ToList(); // Materializa a lista de tensores processados

        // Empilha os tensores para formar o lote final
        var stackedTensor = torch.stack(paddedTensors);

        // Libera os tensores individuais da lista após empilhar
        foreach (var tensor in paddedTensors)
        {
             // Apenas libera se não for o tensor original retornado diretamente no caso else
             if (!batch.Contains(tensor))
             {
                  tensor.Dispose();
             }
        }
        // Libera os tensores originais do batch que foram substituídos por cópias/padded/truncated
        foreach(var t in batch) t.Dispose();


        return stackedTensor.to_type(ScalarType.Int64); // Garante o tipo correto [batch, seq_len]
    }
}
// --- END OF FILE Trainer.cs ---