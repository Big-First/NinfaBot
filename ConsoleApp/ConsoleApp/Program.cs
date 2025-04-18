using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.Encodings.Web;
using System.Threading.Tasks;
using ConsoleApp;

class Program
{
    static async Task Main(string[] args)
    {
        string inputPath = Path.Combine(Directory.GetCurrentDirectory(), "Vocabularys", "tokenizer.json");
        // Let's use a different output name to avoid confusion
        string outputPath = Path.Combine(Directory.GetCurrentDirectory(), "Vocabularys", "tokenizer_bpe.json");

        await GenerateSimulatedBpeMerges(inputPath, outputPath);
    }

    public static async Task GenerateSimulatedBpeMerges(string inputJsonPath, string outputJsonPath)
    {
        TokenWrapper token = null;
        
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
            var tokenizer = JsonSerializer.Deserialize<TokenData>(json, deserializeOptions);
            Console.WriteLine($"{tokenizer == null} deserializacao via Json fail !");

            if (tokenizer?.model?.vocab == null)
            {

                return;
            }

            tokenizer.model.type = "BPE";
            
            Directory.CreateDirectory(Path.GetDirectoryName(outputJsonPath));

            // Serializa com escape mínimo e indentação
            var serializeOptions = new JsonSerializerOptions
            {
                WriteIndented = true,
                Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
                // Keep DefaultIgnoreCondition.WhenWritingNull if you want to omit null fields
                DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
            };
            var model = new Model("BPE", tokenizer.model.vocab, tokenizer.model.merges);
            token = new TokenWrapper("1.0", tokenizer.normalizer, tokenizer.preTokenizer, tokenizer.addedTokens, tokenizer.truncation, tokenizer.padding, tokenizer.postProcessor, tokenizer.decoder,model);

            Console.WriteLine($"💾 Salvando tokenizer atualizado como BPE em: {outputJsonPath}");
            string newJson = JsonSerializer.Serialize(token, serializeOptions);
            Console.WriteLine(newJson);
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
}