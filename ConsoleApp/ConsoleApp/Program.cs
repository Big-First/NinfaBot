using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.Encodings.Web;
using System.Threading.Tasks;

public class TokenizerConfig // Renamed for clarity, can map more fields
{
    [JsonPropertyName("version")]
    public string Version { get; set; }

    [JsonPropertyName("truncation")]
    public object Truncation { get; set; } // Keep original fields if needed

    [JsonPropertyName("padding")]
    public object Padding { get; set; } // Keep original fields if needed

    [JsonPropertyName("added_tokens")]
    public List<AddedToken> AddedTokens { get; set; } // Keep original fields

    [JsonPropertyName("normalizer")]
    public Normalizer Normalizer { get; set; } // Keep original fields

    [JsonPropertyName("pre_tokenizer")]
    public PreTokenizer PreTokenizer { get; set; } // Keep original fields

    [JsonPropertyName("post_processor")]
    public PostProcessor PostProcessor { get; set; } // Keep original fields

    [JsonPropertyName("decoder")]
    public Decoder Decoder { get; set; } // Keep original fields

    [JsonPropertyName("model")]
    public Model Model { get; set; }
}

public class AddedToken
{
    [JsonPropertyName("id")]
    public int Id { get; set; }
    [JsonPropertyName("type")]
    public string Type { get; set; }
    [JsonPropertyName("content")]
    public string Content { get; set; }
    [JsonPropertyName("single_word")]
    public bool SingleWord { get; set; }
    [JsonPropertyName("lstrip")]
    public bool Lstrip { get; set; }
    [JsonPropertyName("rstrip")]
    public bool Rstrip { get; set; }
    [JsonPropertyName("normalized")]
    public bool Normalized { get; set; }
    [JsonPropertyName("special")]
    public bool Special { get; set; }
}

public class Normalizer
{
    [JsonPropertyName("type")]
    public string Type { get; set; }
    [JsonPropertyName("normalizers")]
    public List<NormalizerStep> Normalizers { get; set; }
}
public class NormalizerStep
{
    [JsonPropertyName("type")]
    public string Type { get; set; }
}

public class PreTokenizer
{
    [JsonPropertyName("type")]
    public string Type { get; set; }
}

public class PostProcessor
{
    // Simplified - add full structure if needed for preservation
     [JsonPropertyName("type")]
    public string Type { get; set; }
    // Add other fields like single, pair, special_tokens if needed
}

public class Decoder
{
     [JsonPropertyName("type")]
    public string Type { get; set; }
    [JsonPropertyName("prefix")]
    public string Prefix { get; set; }
     [JsonPropertyName("cleanup")]
    public bool Cleanup { get; set; }
}


public class Model
{
    [JsonPropertyName("type")]
    public string Type { get; set; }

    [JsonPropertyName("unk_token")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string UnkToken { get; set; } // Keep original fields

    [JsonPropertyName("continuing_subword_prefix")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string ContinuingSubwordPrefix { get; set; } // Keep original fields

    [JsonPropertyName("max_input_chars_per_word")]
     [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? MaxInputCharsPerWord { get; set; } // Keep original fields

    [JsonPropertyName("vocab")]
    public Dictionary<string, int> Vocab { get; set; }

    // Changed from List<List<string>> to List<string> for standard BPE merges
    [JsonPropertyName("merges")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] // Only write if not null
    public List<string> Merges { get; set; }
}

class Program
{
    static async Task Main(string[] args)
    {
        string inputPath = Path.Combine(Directory.GetCurrentDirectory(), "Vocabularys", "tokenizer.json");
        // Let's use a different output name to avoid confusion
        string outputPath = Path.Combine(Directory.GetCurrentDirectory(), "Vocabularys", "tokenizer_bpe_simulated.json");

        await GenerateSimulatedBpeMerges(inputPath, outputPath);
    }

    public static async Task GenerateSimulatedBpeMerges(string inputJsonPath, string outputJsonPath)
    {
        try
        {
            // Verifica se o arquivo de entrada existe
            if (!File.Exists(inputJsonPath))
            {
                Console.WriteLine($"❌ Arquivo não encontrado: {inputJsonPath}");
                return;
            }

            Console.WriteLine($"📖 Lendo tokenizer de: {inputJsonPath}");
            string json = await File.ReadAllTextAsync(inputJsonPath);

            Console.WriteLine("📖 Desserializando tokenizer completo...");
            // Use less strict options initially to capture the whole structure
            var deserializeOptions = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            };
            var tokenizer = JsonSerializer.Deserialize<TokenizerConfig>(json, deserializeOptions);

            if (tokenizer?.Model?.Vocab == null)
            {
                Console.WriteLine("❌ Estrutura do JSON inválida ou vocabulário não encontrado.");
                return;
            }

            Console.WriteLine("🔍 Gerando merges BPE simulados a partir do vocabulário WordPiece...");

            var vocab = tokenizer.Model.Vocab;
            var vocabTokens = new HashSet<string>(vocab.Keys); // Faster lookups
            var mergePairs = new HashSet<string>(); // Store unique "token1 token2" strings

            string subwordPrefix = tokenizer.Model.ContinuingSubwordPrefix ?? "##"; // Get prefix from model or default to ##

            foreach (string token1 in vocabTokens)
            {
                // Merge candidate must be a whole word (doesn't start with prefix)
                if (token1.StartsWith(subwordPrefix)) continue;

                foreach (string token2 in vocabTokens)
                {
                    // Merge candidate must be a subword (starts with prefix)
                    if (!token2.StartsWith(subwordPrefix)) continue;

                    // Construct the potential merged token
                    string mergedToken = token1 + token2.Substring(subwordPrefix.Length);

                    // Check if the *result* of the merge also exists in the vocabulary
                    if (vocabTokens.Contains(mergedToken))
                    {
                        // Format follows typical BPE merge lists
                        mergePairs.Add($"{token1} {token2}");
                    }
                }
            }

            var sortedMerges = mergePairs.ToList();
            // Sort alphabetically for deterministic output (real BPE is frequency-based)
            sortedMerges.Sort(StringComparer.Ordinal);

            Console.WriteLine($"✅ {sortedMerges.Count} merges BPE simulados gerados.");

            // Atualiza o modelo no objeto tokenizer
            tokenizer.Model.Type = "BPE"; // Set type to BPE
            tokenizer.Model.Merges = sortedMerges; // Add the generated merges
            // Optional: Clean the vocab here if desired (e.g., remove ##), but be aware of consequences
            // tokenizer.Model.Vocab = CleanVocabForBPE(tokenizer.Model.Vocab, subwordPrefix);
            // tokenizer.Model.ContinuingSubwordPrefix = null; // BPE doesn't typically use this


            // Garante que o diretório de saída existe
            Directory.CreateDirectory(Path.GetDirectoryName(outputJsonPath));

            // Serializa com escape mínimo e indentação
            var serializeOptions = new JsonSerializerOptions
            {
                WriteIndented = true,
                Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
                // Keep DefaultIgnoreCondition.WhenWritingNull if you want to omit null fields
                DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
            };

            Console.WriteLine($"💾 Salvando tokenizer atualizado como BPE em: {outputJsonPath}");
            string newJson = JsonSerializer.Serialize(tokenizer, serializeOptions);
            await File.WriteAllTextAsync(outputJsonPath, newJson);

            Console.WriteLine($"✅ Processo concluído.");
        }
        catch (JsonException ex)
        {
            Console.WriteLine($"❌ Erro ao processar JSON: {ex.Message} (Linha: {ex.LineNumber}, Posição: {ex.BytePositionInLine})");
        }
        catch (IOException ex)
        {
            Console.WriteLine($"❌ Erro de arquivo: {ex.Message}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Erro inesperado: {ex.Message}");
            Console.WriteLine(ex.StackTrace); // More details for unexpected errors
        }
    }

    // Example function if you wanted to try cleaning the vocab (use with caution)
    /*
    public static Dictionary<string, int> CleanVocabForBPE(Dictionary<string, int> wordPieceVocab, string prefix)
    {
        var bpeVocab = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach(var kvp in wordPieceVocab)
        {
            string token = kvp.Key;
            // Decide how to handle subwords, e.g., remove prefix or add space prefix 'Ġ'
            // This is complex and depends on the target BPE model style (e.g., GPT-2, Roberta)
            // Simple example: just remove prefix
            if (token.StartsWith(prefix))
            {
               // What to do? Maybe skip? Or remove prefix?
               // string cleanedToken = token.Substring(prefix.Length);
               // Need careful handling of potential collisions after cleaning!
               // For now, let's just keep the original vocab including ##
               bpeVocab[token] = kvp.Value;
            }
            else
            {
                 bpeVocab[token] = kvp.Value;
            }
        }
        return bpeVocab;
    }
    */
}