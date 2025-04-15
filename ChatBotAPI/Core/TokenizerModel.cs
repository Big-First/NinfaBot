namespace ChatBotAPI.Core;

public class TokenizerModel
{
    public string type { get; set; } = "";
    public Dictionary<string, int> vocab { get; set; } = new Dictionary<string, int>();
}