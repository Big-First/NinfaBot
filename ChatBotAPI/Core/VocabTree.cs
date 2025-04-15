using System.Text.Json;

namespace ChatBotAPI.Core;

// --- ÁRVORE BINÁRIA PARA VOCABULÁRIO ---
public class VocabTree
{
    private class VocabNode
    {
        public int TokenId { get; set; }
        public string Word { get; set; }
        public VocabNode? Left { get; set; }
        public VocabNode? Right { get; set; }

        public VocabNode(int tokenId, string word)
        {
            TokenId = tokenId;
            Word = word;
        }
    }

    private VocabNode? _root;

    public VocabTree(string vocabFile)
    {
        var tokenToId = JsonSerializer.Deserialize<Dictionary<string, int>>(File.ReadAllText(vocabFile))
                        ?? throw new InvalidOperationException("Falha ao carregar vocab.json");

        foreach (var kvp in tokenToId)
        {
            _root = Insert(_root, kvp.Value, kvp.Key);
        }

        Console.WriteLine($"Árvore de vocabulário construída com {tokenToId.Count} nós.");
    }

    private VocabNode Insert(VocabNode? node, int tokenId, string word)
    {
        if (node == null)
            return new VocabNode(tokenId, word);

        if (tokenId < node.TokenId)
            node.Left = Insert(node.Left, tokenId, word);
        else if (tokenId > node.TokenId)
            node.Right = Insert(node.Right, tokenId, word);

        return node;
    }

    public List<int> Encode(string text)
    {
        if (string.IsNullOrEmpty(text)) return new List<int>();
        var tokens = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var ids = new List<int>();
        foreach (var token in tokens)
        {
            int id = FindTokenId(_root, token);
            ids.Add(id >= 0 ? id : 50256); // Fallback para <|endoftext|>
        }

        return ids;
    }

    private int FindTokenId(VocabNode? node, string word)
    {
        if (node == null) return -1;
        if (node.Word == word) return node.TokenId;
        return FindTokenId(node.Left, word) >= 0 ? FindTokenId(node.Left, word) : FindTokenId(node.Right, word);
    }

    public string Decode(List<int> ids)
    {
        if (ids == null || ids.Count == 0) return "";
        var tokens = new List<string>();
        foreach (var id in ids)
        {
            string word = FindWord(_root, id);
            tokens.Add(string.IsNullOrEmpty(word) ? "<unk>" : word);
        }

        return string.Join(" ", tokens);
    }

    private string FindWord(VocabNode? node, int tokenId)
    {
        if (node == null) return "";
        if (node.TokenId == tokenId) return node.Word;
        return tokenId < node.TokenId ? FindWord(node.Left, tokenId) : FindWord(node.Right, tokenId);
    }
}