namespace ChatBotAPI.Core;

public class WorldVocab
{
    public WorldVocab(){}
    public int TokenId { get; set; }
    public string Word { get; set; }
    public WorldVocab? left { get; set; }
    public WorldVocab? right { get; set; }

    public WorldVocab(int tokenId, string word)
    {
        TokenId = tokenId;
        Word = word;
    }
}