using TorchSharp;
using static TorchSharp.torch;
using System; // Para Math
using System.Linq; // Para Linq

namespace LLM.Core;

public static class SamplingUtils
{
    /// <summary>
    /// Amostra um token a partir dos logits usando temperatura, top-k ou top-p.
    /// </summary>
    /// <param name="logits">Logits do último token (shape [vocabSize])</param>
    /// <param name="temperature">Controla a aleatoriedade (valores > 0). 0 para greedy.</param>
    /// <param name="topK">Considera apenas os K tokens mais prováveis (0 para desativar).</param>
    /// <param name="topP">Considera o menor conjunto de tokens cuja probabilidade cumulativa é >= P (0 para desativar).</param>
    /// <returns>ID do token amostrado.</returns>
    // Renomeado de SampleWithSoftmax para SampleNextToken
    public static int SampleNextToken(Tensor logits, double temperature = 0.7, int topK = 0, double topP = 0.0)
    {
        if (logits.dim() != 1)
            throw new ArgumentException("Logits devem ser 1D (vetor de vocabulário).");

        // Não usar DisposeScope dentro de uma função utilitária que retorna um valor.
        // O chamador deve gerenciar o dispose dos tensores retornados ou criados.
        // Mas neste caso, multinomial retorna um tensor, que é convertido para int.
        // Os tensores intermediários (probs, filteredProbs, sortedProbs, etc.)
        // DEVEM ser dispostos dentro desta função.

        // Clona os logits para não modificar o tensor original
        using var tempLogits = logits.clone();
        using var scope = NewDisposeScope(); // Gerenciar tensores INTERMEDIÁRIOS

        // 1. Greedy Sampling (se temperatura for 0 ou muito baixa)
        if (temperature <= 1e-6)
        {
            var result = (int)tempLogits.argmax().item<long>();
            return result;
        }

        // 2. Aplicar Temperatura
        var temperedLogits = tempLogits / temperature; // temperedLogits será gerenciado pelo scope

        // 3. Calcular Probabilidades
        var probs = torch.nn.functional.softmax(temperedLogits, dim: 0); // probs será gerenciado pelo scope

        // 4. Aplicar Top-K ou Top-P
        Tensor filteredProbs;
        if (topK > 0)
        {
            filteredProbs = ApplyTopK(probs, topK); // filteredProbs será gerenciado pelo scope
        }
        else if (topP > 0.0 && topP < 1.0)
        {
            filteredProbs = ApplyTopP(probs, topP); // filteredProbs será gerenciado pelo scope
        }
        else
        {
            // Sem filtragem, usar as probabilidades originais
            // Cria uma cópia para ser gerenciada pelo scope, ou clone() antes do scope?
            // Vamos garantir que filteredProbs sempre seja um tensor criado dentro do scope
            filteredProbs = probs.clone(); // clone() cria um novo tensor, gerenciado pelo scope
        }

        // 5. Amostrar da distribuição resultante
        // multinomial espera probabilidades (soma = 1), não logits
        // Se filteredProbs.sum() for ~0 (pode acontecer com filtragem agressiva e underflow),
        // multinomial pode falhar. Adicionar um epsilon ou fallback para greedy.
        if (filteredProbs.sum().item<double>() < 1e-9)
        {
             Console.WriteLine("AVISO: Probabilidades filtras somam ~0. Fallback para Greedy.");
             var result = (int)tempLogits.argmax().item<long>(); // Usar logits originais (temperados)
             return result;
        }


        // Garantir que a soma é 1 após filtragem (float precision issues)
        // filteredProbs = filteredProbs / filteredProbs.sum(); // ApplyTopK/P já fazem isso

        // A amostragem multinomial pode ter problemas com GPUs se as probabilidades somarem 0.
        // Certificar-se de que não há NaN/Inf após operações
        if (filteredProbs.isnan().any().item<bool>() || filteredProbs.isinf().any().item<bool>())
        {
             Console.WriteLine("AVISO: Probabilidades filtras contêm NaN/Inf. Fallback para Greedy.");
             var result = (int)tempLogits.argmax().item<long>();
             return result;
        }


        var nextTokenTensor = torch.multinomial(filteredProbs, num_samples: 1); // nextTokenTensor será gerenciado pelo scope

        var nextToken = (int)nextTokenTensor.item<long>(); // Obter o valor int

        // O scope.Dispose() será chamado automaticamente ao sair do bloco `using`.
        // Isso liberará todos os tensores intermediários criados dentro do scope.

        return nextToken; // Retorna o int (não um tensor)
    }

    /// <summary>
    /// Filtra probabilidades para manter apenas os K maiores (Top-K).
    /// Zera as outras probabilidades e re-normaliza.
    /// </summary>
    private static Tensor ApplyTopK(Tensor probs, int k)
    {
        // Assume probs está em um DisposeScope ou será disposta pelo chamador.
        // Os tensores criados AQUI precisam ser gerenciados (pelo scope do chamador ou localmente se complexo).
        // Como ApplyTopK/P são chamados DENTRO do scope de SampleNextToken, seus tensores intermediários
        // serão gerenciados por aquele scope. O tensor de retorno (filteredProbs) também será gerenciado por ele.

        // Garante k dentro dos limites e positivo
        k = Math.Max(1, Math.Min(k, (int)probs.shape[0])); // Deve manter pelo menos 1 token se k > 0

        // Obter os k maiores valores e seus índices
        var topKResult = torch.topk(probs, k, dim: 0);
        var topValues = topKResult.values; // Gerenciado pelo scope do chamador via topKResult
        var topIndices = topKResult.indices; // Gerenciado pelo scope do chamador via topKResult

        // Criar um tensor de zeros e preencher as posições top-k com suas probs
        var filteredProbs = torch.full_like(probs, 0.0f); // filteredProbs será gerenciado pelo scope do chamador
        // Index_put_ espera um índice de tensor 1D e os valores correspondentes
        filteredProbs.index_put_(topIndices.unsqueeze(0), topValues.unsqueeze(0)); // precisa de shape [1, k] e [1, k] para index_put_ ou usar overload

        // Re-normalizar para que a soma seja 1
        // Evita divisão por zero se a soma for muito pequena
        var sum = filteredProbs.sum();
        if (sum.item<double>() > 1e-9)
        {
            filteredProbs = filteredProbs / sum;
        }
        // else { filteredProbs continua sendo zeros } - multinomial deve lidar com isso ou usamos fallback

        return filteredProbs; // Retorna o tensor (gerenciado pelo scope do chamador)
    }

    /// <summary>
    /// Filtra probabilidades mantendo o menor conjunto cuja soma cumulativa >= p (Top-P).
    /// Zera as outras probabilidades e re-normaliza.
    /// </summary>
    private static Tensor ApplyTopP(Tensor probs, double p)
    {
        // Assume probs está em um DisposeScope ou será disposta pelo chamador.
         // Os tensores criados AQUI precisam ser gerenciados pelo scope do chamador.

         // Garante p dentro dos limites
         p = Math.Clamp(p, 0.0, 1.0);
         if (p <= 0.0) // Se p=0, remove tudo. Se p=1, remove nada (a menos que haja underflow). p <= 0.0 pode causar problemas.
         {
             // Se p é 0 ou negativo, apenas retorna o mais provável (greedy)
             // Ou zera tudo, forçando multinomial a talvez falhar ou retornar 0?
             // A lógica Top-P geralmente requer p > 0.
             // Vamos retornar um tensor com 0.0f se p <= 0, forçando multinomial a maybe pick 0 or crash.
             // Ou fallback para TopK=1?
             // A implementação padrão em bibliotecas ML é manter pelo menos o token mais provável.
             return ApplyTopK(probs, 1); // Fallback para TopK=1 (essencialmente greedy)
         }
         if (p >= 1.0) return probs; // Se p >= 1, mantém tudo (sem filtragem Top-P)


         // Ordenar probabilidades em ordem decrescente
         var sortedResult = torch.sort(probs, dim: 0, descending: true);
         var sortedProbs = sortedResult.values; // Gerenciado pelo scope do chamador via sortedResult
         var sortedIndices = sortedResult.indices; // Gerenciado pelo scope do chamador via sortedResult


         // Calcular soma cumulativa
         var cumulativeProbs = torch.cumsum(sortedProbs, dim: 0); // Gerenciado pelo scope do chamador

         // Encontrar os índices a manter (aqueles cuja cumulativa > p são removidos)
         // Queremos manter os tokens ATÉ que a soma cumulativa ULTRAPASSE p.
         // O primeiro índice onde cumulativeProbs > p *EXCLUINDO* ele mesmo
         var sortedIndicesToKeepMask = cumulativeProbs <= p;
         // Garantir que pelo menos o token mais provável (índice 0 após sort) seja mantido
         sortedIndicesToKeepMask[0] = true;


         // Criar um tensor de zeros
         var filteredProbs = torch.full_like(probs, 0.0f); // filteredProbs será gerenciado pelo scope do chamador

         // Selecionar os valores das probabilidades que devem ser mantidos
         using var valuesToKeep = sortedProbs.masked_select(sortedIndicesToKeepMask);
         // Selecionar os índices originais correspondentes aos valores a serem mantidos
         using var originalIndicesToKeep = sortedIndices.masked_select(sortedIndicesToKeepMask);

         // Colocar as probabilidades mantidas de volta em suas posições originais no tensor de zeros
         if (valuesToKeep.shape[0] > 0) // Evitar index_put_ com tensor vazio
         {
            filteredProbs.index_put_(originalIndicesToKeep.unsqueeze(0), valuesToKeep.unsqueeze(0)); // Precisa de reshape [1, k] e [1, k]
         }


         // Re-normalizar
         var sum = filteredProbs.sum();
         if (sum.item<double>() > 1e-9)
         {
              filteredProbs = filteredProbs / sum;
         }


         return filteredProbs; // Retorna o tensor (gerenciado pelo scope do chamador)
    }
}
// --- END OF FILE SamplingUtils.cs ---