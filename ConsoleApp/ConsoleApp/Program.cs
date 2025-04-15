using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.Encodings.Web;
using System.Threading.Tasks;

public class TokenWrapper
{
    [JsonPropertyName("version")]
    public string Version { get; set; }

    [JsonPropertyName("model")]
    public Model Model { get; set; }
}

public class Model
{
    [JsonPropertyName("type")]
    public string Type { get; set; }

    [JsonPropertyName("vocab")]
    public Dictionary<string, int> Vocab { get; set; }

    [JsonPropertyName("merges")]
    public List<List<string>> Merges { get; set; }
}

class Program
{
    static async Task Main(string[] args)
    {
        string inputPath = Path.Combine(Directory.GetCurrentDirectory(), "Vocabularys", "tokenizer.json");
        string outputPath = Path.Combine(Directory.GetCurrentDirectory(), "Vocabularys", "tokenizer_bpe_clean.json");

        try
        {
            // Verifica se o arquivo de entrada existe
            if (!File.Exists(inputPath))
            {
                Console.WriteLine($"❌ Arquivo não encontrado: {inputPath}");
                return;
            }

            Console.WriteLine("📖 Lendo tokenizer...");
            string json = await File.ReadAllTextAsync(inputPath);

            Console.WriteLine("📖 Desserializando...");
            var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
            var tokenizer = JsonSerializer.Deserialize<TokenWrapper>(json, options);

            if (tokenizer?.Model?.Vocab == null)
            {
                Console.WriteLine("❌ Estrutura do JSON inválida.");
                return;
            }

            var vocab = tokenizer.Model.Vocab;
            var fixedVocab = new Dictionary<string, int>(StringComparer.Ordinal); // Ignora case e Unicode
            var usedIds = new HashSet<int>();
            int duplicateWords = 0;
            int duplicateIds = 0;
            int nextFreeId = vocab.Values.DefaultIfEmpty(0).Max() + 1;

            foreach (var pair in vocab.OrderBy(p => p.Value)) // Ordena por ID para consistência
            {
                string word = pair.Key;
                int id = pair.Value;

                // Verifica duplicação de palavras/tokens
                if (fixedVocab.ContainsKey(word))
                {
                    Console.WriteLine($"⚠️ Palavra duplicada ignorada: '{word}' (ID original: {id})");
                    duplicateWords++;
                    continue;
                }

                // Verifica duplicação de IDs
                if (!usedIds.Add(id))
                {
                    while (usedIds.Contains(nextFreeId))
                        nextFreeId++;
                    Console.WriteLine($"⚠️ ID duplicado para '{word}' (ID original: {id}), novo ID: {nextFreeId}");
                    id = nextFreeId++;
                    duplicateIds++;
                }

                fixedVocab[word] = id;
            }

            Console.WriteLine($"📊 Resumo: {duplicateWords} palavras duplicadas, {duplicateIds} IDs duplicados corrigidos.");

            // Atualiza o modelo
            tokenizer.Model.Vocab = fixedVocab;
            tokenizer.Model.Type = "bpe";
            tokenizer.Model.Merges ??= new List<List<string>>();

            // Garante que o diretório de saída existe
            Directory.CreateDirectory(Path.GetDirectoryName(outputPath));

            // Serializa com escape mínimo para preservar \u0120
            var serializeOptions = new JsonSerializerOptions
            {
                WriteIndented = true,
                Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
                DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
            };

            string newJson = JsonSerializer.Serialize(tokenizer, serializeOptions);
            await File.WriteAllTextAsync(outputPath, newJson);

            Console.WriteLine($"✅ Tokenizer limpo salvo como: {outputPath} com {fixedVocab.Count} tokens.");
        }
        catch (JsonException ex)
        {
            Console.WriteLine($"❌ Erro ao processar JSON: {ex.Message}");
        }
        catch (IOException ex)
        {
            Console.WriteLine($"❌ Erro de arquivo: {ex.Message}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Erro inesperado: {ex.Message}");
        }
    }
}