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
            long totalTokensInBatch = 0; // Para calcular a perda por token, se desejar

            // Embaralhar o dataset a cada época
            var shuffledDataset = dataset.OrderBy(_ => Guid.NewGuid()).ToList();

            // Processar em lotes (batches)
            for (int i = 0; i < shuffledDataset.Count; i += options.BatchSize)
            {
                var batchItems = shuffledDataset.Skip(i).Take(options.BatchSize).ToList();
                if (!batchItems.Any()) continue;

                // Criar listas para as sequências tokenizadas antes do padding
                var inputSequences = new List<List<int>>();
                var targetSequences = new List<List<int>>();

                foreach (var (inputStr, outputStr) in batchItems)
                {
                    // Tokenizar Input: apenas o texto, sem tokens especiais adicionais
                    var inputTokens = tokenizer.Encode(inputStr, addSpecialTokens: false);

                    // Tokenizar Target: o texto de saída
                    var outputTokens = tokenizer.Encode(outputStr, addSpecialTokens: false);
                    // Adicionar EOS ao final do output para sinalizar o fim da geração
                    outputTokens.Add(tokenizer.GetEosTokenId());

                    // --- Estratégia de Sequência para Treinamento ---
                    // A forma mais comum para pares input/output é concatenar:
                    // [IDs(inputStr), EOS_ID, IDs(outputStr), EOS_ID]
                    // E o target para a loss é essa mesma sequência deslocada um passo à frente.
                    // Ex: Sequência = [A, B, EOS, C, D, EOS]
                    // Input real para o modelo (o que ele "vê" para prever o próximo): [A, B, EOS, C, D, EOS]
                    // Target real para calcular a Loss (o que ele "deveria" prever): [B, EOS, C, D, EOS, PAD]

                    // Criar a sequência completa de treinamento (input + EOS + output + EOS)
                    var fullSequence = new List<int>(inputTokens);
                    fullSequence.Add(tokenizer.GetEosTokenId()); // Separador/Fim do input
                    fullSequence.AddRange(outputTokens); // Adiciona outputTokens (já tem EOS no final)

                    // Truncar a sequência completa se for maior que maxSeqLen do trainer/modelo
                    // O maxSeqLen do trainer DEVE ser <= ao maxSeqLen do modelo.
                    fullSequence = fullSequence.Take(options.MaxSeqLen).ToList();

                    // O input para o modelo é a sequência completa
                    var modelInput = fullSequence;

                    // O target para a loss é a sequência completa deslocada 1 posição para a frente
                    // Remove o primeiro token e adiciona PAD_ID no final para manter o comprimento
                    var targetLoss = fullSequence.Skip(1).ToList();
                    targetLoss.Add(padTokenId);

                    // Truncar o target para o mesmo comprimento do input após truncamento
                    targetLoss = targetLoss.Take(modelInput.Count).ToList();

                    inputSequences.Add(modelInput);
                    targetSequences.Add(targetLoss);
                } // Fim do loop de itens do batch

                // Fazer Padding e criar tensores de lote
                // Usar PaddingHelper.PadSequence para padar as listas de ints
                var paddedInputSequences = inputSequences.Select(seq => PaddingHelper.PadSequence(seq, options.MaxSeqLen, padTokenId)).ToList();
                var paddedTargetSequences = targetSequences.Select(seq => PaddingHelper.PadSequence(seq, options.MaxSeqLen, padTokenId)).ToList();

                // Converter as listas de listas (padded) para tensores TorchSharp long[][]
                var inputTensorArray = paddedInputSequences.Select(seq => seq.Select(id => (long)id).ToArray()).ToArray();
                var targetTensorArray = paddedTargetSequences.Select(seq => seq.Select(id => (long)id).ToArray()).ToArray();

                // Criar tensores TorchSharp a partir dos arrays e mover para o device
                using var batchScope = torch.NewDisposeScope(); // Gerenciar memória do lote
                var inputBatchTensor = torch.tensor(inputTensorArray, dtype: ScalarType.Int64, device: device);
                var targetBatchTensor = torch.tensor(targetTensorArray, dtype: ScalarType.Int64, device: device);

                // Etapa de Treinamento
                optimizer.zero_grad(); // Limpar gradientes anteriores

                // Forward pass. model(inputBatchTensor) => Logits [B, S, V]
                // O modelo já lida com o truncamento interno se options.MaxSeqLen > model.MaxSeqLen
                // Passamos options.MaxSeqLen aqui para o padding.
                var logits = model.forward(inputBatchTensor);

                // Calcular Loss
                // Logits precisam ser [B*S, V] para CrossEntropyLoss
                // Targets precisam ser [B*S] para CrossEntropyLoss
                // targetBatchTensor já está em [B, S], view(-1) transforma em [B*S]
                var loss = lossFn.forward(logits.view(-1, model.VocabSize), targetBatchTensor.view(-1)); // Usar model.VocabSize

                // Verificar se a loss é NaN ou Inf (pode acontecer com LR muito alto ou dados ruins)
                if (loss.isnan().any().item<bool>() || loss.isinf().any().item<bool>())
                {
                    Console.WriteLine($"AVISO: Loss NaN/Inf na época {epoch}, lote {batchCount}. Ignorando este lote ou parando.");
                    // Você pode decidir pular este lote (continue;) ou parar o treinamento (break;)
                    // continue; // Pula o lote
                    break; // Para o treinamento
                }


                loss.backward(); // Calcular gradientes
                optimizer.step(); // Atualizar pesos

                totalLoss += loss.item<double>();
                batchCount++;
                totalTokensInBatch += inputBatchTensor.shape[0] * inputBatchTensor.shape[1]; // Contar tokens no lote PADADO


                // Dispor tensores do lote explicitamente (ajuda GC e memória GPU)
                // O batchScope já faz isso no final do using block
                inputBatchTensor.Dispose();
                targetBatchTensor.Dispose();
                logits.Dispose();
                loss.Dispose();
                 batchScope.Dispose(); // Garante que tudo criado no scope é liberado

            } // Fim do loop de lotes

            // Se o treinamento parou por causa de NaN/Inf loss
            if (double.IsNaN(totalLoss) || double.IsInfinity(totalLoss)) break;


            double avgLoss = totalLoss / batchCount;
            Console.WriteLine($"📚 Época {epoch}/{options.Epochs} - Loss Média: {avgLoss:F4}");

            // Opcional: Salvar o modelo a cada N épocas
            // if (epoch % 5 == 0) // Salvar a cada 5 épocas
            // {
            //     try
            //     {
            //         model.Save(options.SavePath);
            //         Console.WriteLine($"💾 Modelo salvo após época {epoch} em: {options.SavePath}");
            //     }
            //     catch (Exception ex)
            //     {
            //         Console.WriteLine($"❌ Erro ao salvar o modelo após época {epoch}: {ex.Message}");
            //     }
            // }


        } // Fim do loop de épocas

        Console.WriteLine("✅ Treinamento finalizado.");
        try
        {
            // Salvar o modelo no final do treinamento
            model.Save(options.SavePath);
            Console.WriteLine($"💾 Modelo salvo em: {options.SavePath}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Erro ao salvar o modelo: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }

    /// <summary>
    /// Método auxiliar para padar sequências e criar tensor batch, movendo para o device.
    /// Refatorado para ser mais simples e usar o PaddingHelper.
    /// </summary>
    private Tensor PadBatchAndMove(List<Tensor> batch, int seqLen, int padValue, Device device)
    {
        // Esta função não é mais usada pois a lógica de padding foi movida
        // para a criação das listas de int (paddedInputSequences, paddedTargetSequences)
        // e a criação dos tensores a partir delas.
        // Pode remover esta função morta.
        throw new NotImplementedException("PadBatchAndMove não é mais usado diretamente no Trainer.");
    }
}
// --- END OF FILE Trainer.cs ---