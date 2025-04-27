using System.Text;
using System.Text.RegularExpressions;
using AI.Core;

namespace AI.Core
{
    /// <summary>
    /// Serviço para realizar resumo de texto extrativo simples.
    /// </summary>
    /// <remarks>
    /// Este serviço seleciona frases do texto original até atingir um limite de tokens.
    /// Ele não utiliza o modelo de linguagem neural para a sumarização.
    /// </remarks>
    public class SummarizationService
    {
        private readonly Tokenizer tokenizer;

        /// <summary>
        /// Inicializa uma nova instância do SummarizationService.
        /// </summary>
        /// <param name="tokenizer">O tokenizador, usado aqui para contar tokens das frases.</param>
        /// <exception cref="ArgumentNullException">Lançada se o tokenizador for nulo.</exception>
        public SummarizationService(Tokenizer tokenizer)
        {
            this.tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
            Console.WriteLine("SummarizationService initialized.");
        }

        /// <summary>
        /// Realiza um resumo extrativo simples de um texto.
        /// </summary>
        /// <param name="text">O texto de entrada para resumir.</param>
        /// <param name="maxSummaryTokens">O número máximo aproximado de tokens para o resumo.</param>
        /// <returns>O texto resumido.</returns>
        /// <remarks>
        /// Este método quebra o texto em frases e adiciona frases ao resumo sequencialmente
        /// até que o limite de tokens seja atingido ou todas as frases sejam adicionadas.
        /// </remarks>
        public string SummarizeText(string text, int maxSummaryTokens)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                return "[Texto de entrada vazio ou nulo]";
            }
            if (maxSummaryTokens <= 0)
            {
                 return "[Limite de tokens para resumo deve ser positivo]";
            }

            Console.WriteLine($"Summarizing text (approx {text.Length} chars) to max {maxSummaryTokens} tokens...");

            // Quebrar texto em frases - Regex simples, pode não ser perfeito para todos os casos
            // Este Regex tenta quebrar por ., !, ? seguidos por espaço ou fim de string
            var sentences = Regex.Split(text, @"(?<=[\.\!\?])\s+", RegexOptions.Multiline)
                               .Where(s => !string.IsNullOrWhiteSpace(s))
                               .ToList();

            if (!sentences.Any())
            {
                 Console.WriteLine("No sentences found in the text.");
                return "[Não foi possível encontrar frases no texto para resumir]";
            }

            StringBuilder summary = new StringBuilder();
            int currentTokenCount = 0;
            List<string> summarySentences = new List<string>();

            foreach (var sentence in sentences)
            {
                // Tokeniza a frase para contar tokens (sem padding/truncamento)
                int[] sentenceTokens = tokenizer.Tokenize(sentence, applyPaddingTruncation: false);
                int sentenceTokenCount = sentenceTokens.Length;

                // Verifica se adicionar esta frase excederá o limite
                if (currentTokenCount + sentenceTokenCount > maxSummaryTokens)
                {
                    // Se for a primeira frase e já exceder o limite, adiciona-a mesmo assim
                    // para evitar resumo vazio, mas a trunca
                    if (summarySentences.Count == 0)
                    {
                         // Tentativa simples de truncar a primeira frase longa
                         // Pega tokens até o limite e destokeniza
                         var truncatedTokens = sentenceTokens.Take(maxSummaryTokens).ToArray();
                         string truncatedSentence = tokenizer.Detokenize(truncatedTokens);
                         summarySentences.Add(truncatedSentence.Trim());
                         currentTokenCount += truncatedTokens.Length;
                         Console.WriteLine($"Added truncated first sentence ({truncatedTokens.Length} tokens).");
                         break; // Sai após adicionar a primeira frase truncada
                    }
                    // Caso contrário, para se a próxima frase exceder o limite
                    Console.WriteLine($"Stopping summary: Next sentence ({sentenceTokenCount} tokens) exceeds max limit ({maxSummaryTokens - currentTokenCount} remaining).");
                    break;
                }

                // Adiciona a frase completa
                summarySentences.Add(sentence.Trim()); // Adiciona a frase original (sem trim excessivo)
                currentTokenCount += sentenceTokenCount;
                Console.WriteLine($"Added sentence ({sentenceTokenCount} tokens). Current total: {currentTokenCount}/{maxSummaryTokens}");
            }

            // Junta as frases selecionadas
            string finalSummary = string.Join(" ", summarySentences);

             Console.WriteLine($"Summarization finished. Final token count: {currentTokenCount}.");

            return finalSummary;
        }
    }
}