using NotImplementedException = System.NotImplementedException;
using TorchSharp;
using static TorchSharp.torch; // Necessário para tensor, arange, etc.
using static TorchSharp.torch.optim; // Necessário para Adam
using static TorchSharp.torch.nn; // Necessário para CrossEntropyLoss, Module
using System; // Para Guid, Math
using System.Collections.Generic; // Para List, Dictionary
using System.Linq; // Para Linq (Select, ToList, Skip, Take, Sum)
using LLM.Core.Utils;
using TorchSharp.Modules; // Adicionar using para PaddingHelper

namespace LLM.Core;

public class Trainer
{
    private readonly TransformerModel model;
    private readonly Tokenizer tokenizer;
    private readonly TrainerOptions options;
    private readonly Device device;
    private readonly CrossEntropyLoss lossFn; // Especificar tipo para acesso a ignore_index
    private readonly Optimizer optimizer;
    private readonly int padTokenId;

    public Trainer(TransformerModel model, Tokenizer tokenizer, TrainerOptions options)
    {
        this.model = model;
        this.tokenizer = tokenizer;
        this.options = options;
        // CORREÇÃO: Usar o mesmo dispositivo que o modelo já está (definido em Startup.cs)
        this.device = model.device;
        this.padTokenId = tokenizer.GetPadTokenId(); // Obter ID de padding do tokenizer

        // CrossEntropyLoss para classificação (predizer o próximo token)
        // ignore_index faz com que a loss seja 0 para os tokens de padding no target
        // CORREÇÃO: Instanciar CrossEntropyLoss usando o método de fábrica nn.CrossEntropyLoss
        // e passar o parâmetro ignore_index.
        this.lossFn = CrossEntropyLoss(ignore_index: padTokenId);

        // O otimizador AdamW é comum em Transformers, mas Adam também funciona.
        // Certifique-se de que o otimizador está sendo criado APÓS o modelo ser movido para o device.
        // Como o modelo é movido para o device no Startup antes de o Trainer ser criado, isso está correto.
        // CORREÇÃO: Instanciar Adam usando o método de fábrica torch.optim.Adam
        this.optimizer = Adam(model.parameters(), lr: options.LearningRate);


        // model.to(device); // O modelo JÁ DEVE estar no device correto vindo do DI em Startup.cs
        Console.WriteLine($"Treinamento será executado em: {device.type}");
        if (padTokenId != -1) // Verificar se PAD_ID foi encontrado
        {
             Console.WriteLine($"   CrossEntropyLoss ignorará o token ID: {padTokenId}");
        }
        else
        {
             Console.WriteLine("   AVISO: PAD Token ID não encontrado ou inválido. CrossEntropyLoss pode não ignorar padding corretamente.");
        }
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
            long totalTokensInBatch = 0;

            var shuffledDataset = dataset.OrderBy(_ => Guid.NewGuid()).ToList();

            for (int i = 0; i < shuffledDataset.Count; i += options.BatchSize)
            {
                var batchItems = shuffledDataset.Skip(i).Take(options.BatchSize).ToList();
                if (!batchItems.Any()) continue;

                var inputSequences = new List<List<int>>();
                var targetSequences = new List<List<int>>();

                foreach (var (inputStr, outputStr) in batchItems)
                {
                    var inputTokens = tokenizer.Encode(inputStr, allowSpecialTokensInText: false);
                    var outputTokens = tokenizer.Encode(outputStr, allowSpecialTokensInText: false);
                    outputTokens.Add(tokenizer.GetEosTokenId());

                    var fullSequence = new List<int>(inputTokens);
                    fullSequence.Add(tokenizer.GetEosTokenId()); // Separador/Fim do input
                    fullSequence.AddRange(outputTokens);

                    fullSequence = fullSequence.Take(options.MaxSeqLen).ToList();

                    var modelInput = fullSequence;
                    var targetLoss = fullSequence.Skip(1).ToList();
                    targetLoss.Add(padTokenId);
                    targetLoss = targetLoss.Take(modelInput.Count).ToList();

                    inputSequences.Add(modelInput);
                    targetSequences.Add(targetLoss);
                } // Fim do loop de itens do batch

                // Fazer Padding usando PaddingHelper
                var paddedInputSequences = inputSequences.Select(seq => PaddingHelper.PadSequence(seq, options.MaxSeqLen, padTokenId)).ToList();
                var paddedTargetSequences = targetSequences.Select(seq => PaddingHelper.PadSequence(seq, options.MaxSeqLen, padTokenId)).ToList();

                // --- CORREÇÃO AQUI: Converter List<List<int>> para long[,] ---
                int currentBatchSize = paddedInputSequences.Count;
                int currentSeqLen = currentBatchSize > 0 ? paddedInputSequences[0].Count : 0;

                // Criar arrays 2D
                long[,] inputTensorArray = new long[currentBatchSize, currentSeqLen];
                long[,] targetTensorArray = new long[currentBatchSize, currentSeqLen];

                // Copiar dados das List<List<int>> para os arrays 2D
                for (int b = 0; b < currentBatchSize; b++)
                {
                    for (int s = 0; s < currentSeqLen; s++)
                    {
                        inputTensorArray[b, s] = paddedInputSequences[b][s];
                        targetTensorArray[b, s] = paddedTargetSequences[b][s];
                    }
                }
                // --- Fim da CORREÇÃO ---


                // Criar tensores TorchSharp a partir dos arrays 2D e mover para o device
                // Esta chamada deve encontrar uma sobrecarga compatível com long[,]
                using var batchScope = torch.NewDisposeScope(); // Gerenciar memória do lote
                var inputBatchTensor = torch.tensor(inputTensorArray, dtype: ScalarType.Int64, device: device);
                var targetBatchTensor = torch.tensor(targetTensorArray, dtype: ScalarType.Int64, device: device);


                // Etapa de Treinamento
                optimizer.zero_grad(); // Limpar gradientes anteriores

                var logits = model.forward(inputBatchTensor); // [B, S, V]

                // Calcular Loss
                var loss = lossFn.forward(logits.view(-1, model.VocabSize), targetBatchTensor.view(-1));

                if (loss.isnan().any().item<bool>() || loss.isinf().any().item<bool>())
                {
                    Console.WriteLine($"AVISO: Loss NaN/Inf na época {epoch}, lote {batchCount}. Parando treinamento.");
                    // Dispor tensores antes de parar
                    inputBatchTensor.Dispose();
                    targetBatchTensor.Dispose();
                    logits.Dispose();
                    loss.Dispose();
                    batchScope.Dispose();
                    goto EndTrainingLoops; // Sair de ambos os loops
                }

                loss.backward(); // Calcular gradientes
                optimizer.step(); // Atualizar pesos

                totalLoss += loss.item<double>();
                batchCount++;
                totalTokensInBatch += inputBatchTensor.shape[0] * inputBatchTensor.shape[1];

                // Dispor tensores do lote explicitamente
                inputBatchTensor.Dispose();
                targetBatchTensor.Dispose();
                logits.Dispose();
                loss.Dispose();
                 batchScope.Dispose();
            } // Fim do loop de lotes

            // Se a loss se tornou NaN ou Inf, parar também o loop de época
            // Este check agora está incluído no goto
            // if (double.IsNaN(totalLoss) || double.IsInfinity(totalLoss)) break;


            double avgLoss = totalLoss / batchCount;
            Console.WriteLine($"📚 Época {epoch}/{options.Epochs} - Loss Média: {avgLoss:F4}");

        } // Fim do loop de épocas

    EndTrainingLoops: // Label para sair com goto

        Console.WriteLine("✅ Treinamento finalizado.");
        try
        {
            model.Save(options.SavePath);
            Console.WriteLine($"💾 Modelo salvo em: {options.SavePath}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Erro ao salvar o modelo: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }
}
// --- END OF FILE Trainer.cs ---