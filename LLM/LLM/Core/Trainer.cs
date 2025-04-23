using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.nn;
using System; // Para Guid
using System.Collections.Generic; // Para List
using System.Linq; // Para Linq
using LLM.Core.Utils; // Adicionar using para PaddingHelper

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
        this.device = model.device; // Usar o mesmo dispositivo que o modelo já está
        this.padTokenId = tokenizer.GetPadTokenId(); // Obter ID de padding do tokenizer

        // CrossEntropyLoss para classificação (predizer o próximo token)
        // ignore_index faz com que a loss seja 0 para os tokens de padding no target
        this.lossFn = CrossEntropyLoss(ignore_index: padTokenId);

        // O otimizador AdamW é comum em Transformers, mas Adam também funciona.
        // Certifique-se de que o otimizador está sendo criado APÓS o modelo ser movido para o device.
        // Como o modelo é movido para o device no Startup antes de o Trainer ser criado, isso está correto.
        this.optimizer = torch.optim.Adam(model.parameters(), lr: options.LearningRate);


        // model.to(device); // O modelo JÁ DEVE estar no device correto vindo do DI em Startup.cs
        Console.WriteLine($"Treinamento será executado em: {device.type}");
        if (padTokenId != -1) // Verificar se PAD_ID foi encontrado
        {
             Console.WriteLine($"   CrossEntropyLoss ignorará o token ID: {padTokenId}");
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
            long totalTokensProcessed = 0; // Para calcular tokens/segundo ou outros métricas

            // Embaralhar o dataset a cada época
            var shuffledDataset = dataset.OrderBy(_ => Guid.NewGuid()).ToList();

            // Processar em lotes (batches)
            for (int i = 0; i < shuffledDataset.Count; i += options.BatchSize)
            {
                var batchItems = shuffledDataset.Skip(i).Take(options.BatchSize).ToList();
                if (!batchItems.Any()) continue;

                // Criar listas para os tensores de input e target antes do padding
                var inputTensors = new List<Tensor>();
                var targetTensors = new List<Tensor>(); // Os targets são os tokens de saída deslocados

                foreach (var (inputStr, outputStr) in batchItems)
                {
                    // Tokenizar Input: apenas o texto, sem tokens especiais adicionais
                    var inputTokens = tokenizer.Encode(inputStr, addSpecialTokens: false);

                    // Tokenizar Target: o texto de saída
                    var targetTokens = tokenizer.Encode(outputStr, addSpecialTokens: false);
                    targetTokens.Add(tokenizer.GetEosTokenId()); // Adicionar EOS ao final do target

                    // No treinamento, o input e o target são geralmente deslocados.
                    // O input é a sequência até o penúltimo token, e o target é a sequência
                    // do segundo token até o último (incluindo o EOS).
                    // Ex: Input: [tok1, tok2, tok3], Target: [tok2, tok3, EOS]
                    // O modelo prediz o próximo token com base nos anteriores.
                    // Para prever tok2, ele vê tok1. Para prever tok3, ele vê [tok1, tok2].
                    // Para prever EOS, ele vê [tok1, tok2, tok3].

                    // Se quisermos que o modelo aprenda a completar (input -> output),
                    // a abordagem comum é concatenar input + separador (opcional) + output + EOS,
                    // e usar essa sequência como target (deslocado) para calcular a loss,
                    // mascarando a loss para os tokens do input original.
                    // OU, como no código original, usar input para o forward,
                    // e target como o que deveria sair. Isso implica que o input
                    // deve ser o "prefixo" e o target o "sufixo" a ser gerado.
                    // O código atual em Startup.cs nos endpoints /train e /reforco
                    // usa inputStr e outputStr separadamente. Vamos manter essa lógica
                    // por enquanto, mas a abordagem concatenada é mais comum.
                    // Ajustando a lógica para usar inputStr e outputStr como pares:
                    // O INPUT para o modelo será inputStr. O TARGET será outputStr + EOS.
                    // Precisamos alinhar seus comprimentos para a perda.
                    // O target para a loss é a sequência esperada DESLOCADA.
                    // Se input = "hello", output = "world", target = "world" + EOS
                    // Sequência completa para loss: [IDs("hello"), IDs("world"), EOS_ID]
                    // Input para modelo: [IDs("hello"), IDs("world")]
                    // Target para loss:  [IDs("world"), EOS_ID]
                    // A forma mais simples com a arquitetura decoder-only é usar
                    // toda a sequência como entrada, e o target é essa mesma sequência deslocada.
                    // Ex: seq = [tok1, tok2, tok3, tok4]
                    // Input para modelo: [tok1, tok2, tok3, tok4]
                    // Target para loss:  [tok2, tok3, tok4, PAD] (ou PAD se for EOS no tok4)
                    // OU, se for input/output par: [tok_in_1... tok_in_N, tok_out_1... tok_out_M, EOS]
                    // Input para modelo: [tok_in_1... tok_in_N, tok_out_1... tok_out_M, EOS]
                    // Target para loss:  [tok_in_2... tok_in_N, tok_out_1... tok_out_M, EOS, PAD]
                    // E mascarar a loss para os tokens de input originais.

                    // Vamos seguir a lógica do código original para o par input/output:
                    // Input tensor = IDs do inputStr
                    // Target tensor = IDs do outputStr + EOS_ID
                    // O modelo forward(Input tensor) => Logits [1, seq_len_input, vocab_size]
                    // A loss é calculada comparando os logits do *último* token de input
                    // com o *primeiro* token de target, logits do penúltimo de input com
                    // o último de input (?? Isso não faz sentido).
                    // A loss sempre compara logits[i] com target_token[i].
                    // Se input [A, B], target [C, D], loss(model([A,B])) vs [C,D] não funciona.
                    // A Loss deve ser calculada sobre a sequência completa (input+output)
                    // onde o target para uma posição 'i' é o token na posição 'i+1'.

                    // Vamos ajustar a lógica do Trainer para a forma padrão:
                    // Sequência de Treinamento: [IDs(inputStr), EOS_ID, IDs(outputStr), EOS_ID] ? Ou um token separador?
                    // Um separador tipo "<|sep|>" é comum. GPT-2 usava EOS.
                    // Vamos usar [IDs(inputStr), EOS_ID, IDs(outputStr), EOS_ID] como sequência de treino.
                    // O target para a loss será essa sequência DESLOCADA.

                    // Criar a sequência completa de treinamento
                    var fullSequence = new List<int>(tokenizer.Encode(inputStr, addSpecialTokens: false));
                    fullSequence.Add(tokenizer.GetEosTokenId()); // Separador/Fim do input
                    fullSequence.AddRange(tokenizer.Encode(outputStr, addSpecialTokens: false));
                    fullSequence.Add(tokenizer.GetEosTokenId()); // Fim do output (e da sequência)

                    // Truncar a sequência completa para maxSeqLen do trainer/modelo
                    fullSequence = fullSequence.Take(options.MaxSeqLen).ToList();

                    // O input para o modelo é a sequência completa
                    var modelInput = fullSequence;

                    // O target para a loss é a sequência completa deslocada 1 posição para a frente
                    // O último token do target será o PAD_ID ou será truncado
                    var targetLoss = fullSequence.Skip(1).ToList(); // Skip o primeiro token

                    // Adicionar PAD_ID ao final do target para manter o mesmo comprimento que o input (antes do padding)
                    while (targetLoss.Count < modelInput.Count)
                    {
                        targetLoss.Add(padTokenId);
                    }

                     // Truncar o target para o mesmo comprimento que o input (após truncamento)
                     targetLoss = targetLoss.Take(modelInput.Count).ToList();


                    // Fazer Padding para o tamanho fixo options.MaxSeqLen (se modelInput.Count < options.MaxSeqLen)
                    // Nota: O forward do modelo já lida com input < maxSeqLen e > maxSeqLen (truncando).
                    // Vamos padar AQUI para o tamanho do batch para simplificar a criação do tensor.
                    // Se o maxSeqLen do modelo é 64, mas o batch tem sequências de 30 e 40,
                    // precisamos padá-las para o mesmo comprimento dentro do batch (ex: 40 se max_batch_len=40, ou 64).
                    // A função PadBatch faz isso para o max_batch_len ou um fixo.
                    // Vamos padar para options.MaxSeqLen.

                    var paddedModelInput = PaddingHelper.PadSequence(modelInput, options.MaxSeqLen, padTokenId);
                    var paddedTargetLoss = PaddingHelper.PadSequence(targetLoss, options.MaxSeqLen, padTokenId);


                    batchInputs.Add(torch.tensor(paddedModelInput.Select(id => (long)id).ToArray(), dtype: ScalarType.Int64));
                    batchTargets.Add(torch.tensor(paddedTargetLoss.Select(id => (long)id).ToArray(), dtype: ScalarType.Int64));

                } // Fim do loop de itens do batch

                // Fazer Stack dos tensores para formar o lote
                // PadBatch já faz stack e move para o device
                using var batchScope = torch.NewDisposeScope(); // Gerenciar memória do lote
                var inputBatchTensor = PadBatchAndMove(batchInputs, options.MaxSeqLen, padTokenId, device);
                var targetBatchTensor = PadBatchAndMove(batchTargets, options.MaxSeqLen, padTokenId, device);

                // Etapa de Treinamento
                optimizer.zero_grad(); // Limpar gradientes anteriores

                // Forward pass. model(inputBatchTensor) [B, S, V]
                var logits = model.forward(inputBatchTensor);

                // Calcular Loss
                // Logits precisam ser [B*S, V]
                // Targets precisam ser [B*S]
                // targetBatchTensor já está em [B, S], view(-1) transforma em [B*S]
                var loss = lossFn.forward(logits.view(-1, model.vocabSize), targetBatchTensor.view(-1));

                loss.backward(); // Calcular gradientes
                optimizer.step(); // Atualizar pesos

                totalLoss += loss.item<double>();
                batchCount++;
                totalTokensProcessed += batchItems.Sum(item => Math.Min(item.input.Length + item.output.Length + 2, options.MaxSeqLen)); // Estimar tokens processados


                // Dispor tensores do lote explicitamente (ajuda GC e memória GPU)
                // PadBatchAndMove já dispoe os tensores individuais na lista.
                inputBatchTensor.Dispose();
                targetBatchTensor.Dispose();
                logits.Dispose();
                loss.Dispose();
                 batchScope.Dispose(); // Garante que tudo criado no scope é liberado

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
            Console.WriteLine(ex.StackTrace);
        }
    }

    /// <summary>
    /// Adiciona padding a uma lista de tensores para formar um lote, e move para o dispositivo.
    /// Modificado para usar PadSequence do PaddingHelper.
    /// </summary>
    private Tensor PadBatchAndMove(List<Tensor> batch, int seqLen, int padValue, Device device)
    {
        // Dispor os tensores originais na lista antes de processar
        foreach(var t in batch) t.Dispose();

        // Assume que os tensores na lista 'batch' já representam as sequências individuais
        // e suas formas [seq_len] são variadas.
        // Precisamos convertê-los de volta para List<int> para usar PadSequence
        var sequencesAsLists = batch.Select(t => t.data<long>().Take(t.shape[0]).Select(id => (int)id).ToList()).ToList();

        var paddedTensors = sequencesAsLists.Select(seq =>
        {
             // Usa a função PadSequence para padar/truncar cada sequência
            var paddedSeq = PaddingHelper.PadSequence(seq, seqLen, padValue);
            // Cria um novo tensor a partir da sequência paddada e move para o device
            return torch.tensor(paddedSeq.Select(id => (long)id).ToArray(), dtype: ScalarType.Int64).to(device);
        }).ToList(); // Materializa a lista de tensores processados

        if (!paddedTensors.Any()) return torch.empty(0, seqLen, dtype: ScalarType.Int64, device: device); // Retorna um tensor vazio se o lote estiver vazio

        // Empilha os tensores para formar o lote final [batch_size, seq_len]
        var stackedTensor = torch.stack(paddedTensors);

        // Libera os tensores individuais da lista após empilhar
        foreach (var tensor in paddedTensors)
        {
             tensor.Dispose();
        }

        return stackedTensor;
    }

    // Remover a função PadBatch original não modificada
    // /// <summary>
    // /// Adiciona padding a uma lista de tensores para formar um lote.
    // /// </summary>
    // private Tensor PadBatch(List<Tensor> batch, int seqLen, int padValue)
    // {
    //     // ... código original ...
    // }
}
// --- END OF FILE Trainer.cs ---