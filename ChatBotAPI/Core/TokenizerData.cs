namespace ChatBotAPI.Core;

// Classes auxiliares para desserializar tokenizer.json
public class TokenizerData
{
    public string version { get; set; } = "";
    public TokenizerModel model { get; set; } = new TokenizerModel();
}