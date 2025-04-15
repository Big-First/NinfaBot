using System;
using System.IO;
using System.Text.Json;

public class VocabTree
{
    public Dictionary<string, int> Vocab { get; private set; }

    public VocabTree(string filePath)
    {
        LoadVocab(filePath);
    }

    private void LoadVocab(string filePath)
    {
        try
        {
            string jsonContent = File.ReadAllText(filePath);
            var jsonDoc = JsonDocument.Parse(jsonContent);
            var vocabElement = jsonDoc.RootElement.GetProperty("Model").GetProperty("Vocab");

            Vocab = new Dictionary<string, int>();
            foreach (var entry in vocabElement.EnumerateObject())
            {
                Vocab[entry.Name] = entry.Value.GetInt32();
            }

            Console.WriteLine($"Vocabulário carregado com sucesso: {Vocab.Count} entradas.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Erro ao carregar o vocabulário: {ex.Message}");
            Vocab = new Dictionary<string, int> { { "<unk>", 0 } }; // Fallback mínimo
        }
    }
}