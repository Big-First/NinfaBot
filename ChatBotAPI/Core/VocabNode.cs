namespace ChatBotAPI.Core;

// --- ÁRVORE BINÁRIA PARA VOCABULÁRIO ---
public class VocabNode
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