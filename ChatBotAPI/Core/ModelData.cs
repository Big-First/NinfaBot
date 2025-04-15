namespace ChatBotAPI.Core;

public class ModelData
{
    public string type { get; set; } = "";
    public Dictionary<string, int> vocab { get; set; } = new Dictionary<string, int>();
}