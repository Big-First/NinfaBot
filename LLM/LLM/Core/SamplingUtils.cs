// --- START OF FILE SamplingUtils.cs ---
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
    public static int SampleNextToken(Tensor logits, double temperature = 0.7, int topK = 0, double topP = 0.0)
    {
        if (logits.dim() != 1)
            throw new ArgumentException("Logits devem ser 1D (vetor de vocabulário).");

        using var scope = NewDisposeScope(); // Gerencia memória do TorchSharp

        // 1. Greedy Sampling (se temperatura for 0 ou muito baixa)
        if (temperature <= 1e-6)
        {
            return (int)logits.argmax().item<long>();
        }

        // 2. Aplicar Temperatura
        logits = logits / temperature;

        // 3. Calcular Probabilidades
        var probs = torch.nn.functional.softmax(logits, dim: 0);

        // 4. Aplicar Top-K (se habilitado)
        if (topK > 0 && topK < probs.shape[0])
        {
            probs = ApplyTopK(probs, topK);
        }
        // 5. Aplicar Top-P (Nucleus Sampling) (se habilitado e Top-K não foi usado)
        else if (topP > 0.0 && topP < 1.0)
        {
            probs = ApplyTopP(probs, topP);
        }

        // 6. Amostrar da distribuição resultante
        // multinomial espera probabilidades, não logits
        var nextToken = torch.multinomial(probs, num_samples: 1);

        return (int)nextToken.item<long>(); // Retorna como int
    }

    /// <summary>
    /// Filtra probabilidades para manter apenas os K maiores (Top-K).
    /// Zera as outras probabilidades e re-normaliza.
    /// </summary>
    private static Tensor ApplyTopK(Tensor probs, int k)
    {
        // Garante k dentro dos limites
        k = Math.Min(k, (int)probs.shape[0]);
        if (k <= 0) return probs; // Se k for inválido, retorna original

        // Obter os k maiores valores e seus índices
        var (topValues, topIndices) = torch.topk(probs, k, dim: 0);

        // Criar um tensor de zeros e preencher as posições top-k com suas probs
        var filteredProbs = torch.full_like(probs, 0.0f); // Usar float para probs
        filteredProbs.index_put_(topValues, topIndices);

        // Re-normalizar para que a soma seja 1
        return filteredProbs / filteredProbs.sum();
    }

    /// <summary>
    /// Filtra probabilidades mantendo o menor conjunto cuja soma cumulativa >= p (Top-P).
    /// Zera as outras probabilidades e re-normaliza.
    /// </summary>
    private static Tensor ApplyTopP(Tensor probs, double p)
    {
         // Garante p dentro dos limites
         p = Math.Clamp(p, 0.0, 1.0);
         if (p <= 0.0 || p >= 1.0) return probs; // Se p for inválido/trivial, retorna original

         // Ordenar probabilidades em ordem decrescente
         var (sortedProbs, sortedIndices) = torch.sort(probs, dim: 0, descending: true);

         // Calcular soma cumulativa
         var cumulativeProbs = torch.cumsum(sortedProbs, dim: 0);

         // Encontrar os índices a manter (aqueles cuja cumulativa > p são removidos)
         // Adicionamos o primeiro elemento (o mais provável) sempre.
         var sortedIndicesToRemove = cumulativeProbs > p;
         // Desloca para a direita, garantindo que o primeiro elemento nunca seja True
         sortedIndicesToRemove[1..] = sortedIndicesToRemove[..^1].clone();
         sortedIndicesToRemove[0] = false; // Garante que pelo menos o mais provável fique

         // Criar um tensor de zeros e preencher as posições mantidas
         var filteredProbs = torch.full_like(probs, 0.0f);
         // Obter os índices originais a serem zerados
         var indicesToRemove = sortedIndices.masked_select(sortedIndicesToRemove);

         // Copia as probabilidades originais e zera as que devem ser removidas
         // É mais fácil copiar tudo e depois zerar do que preencher a partir de zeros neste caso
         filteredProbs = probs.clone(); // Começa com todas as probs
         filteredProbs.index_fill_(0, indicesToRemove, 0.0f); // Zera as probs a remover

         // Re-normalizar
         return filteredProbs / filteredProbs.sum();
    }
}
// --- END OF FILE SamplingUtils.cs ---