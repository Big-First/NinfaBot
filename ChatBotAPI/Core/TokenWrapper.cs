using System.Text.Json.Serialization;

namespace ChatBotAPI.Core;

public class TokenWrapper
{
    public string version { get; set; }
    public Model model { get; set; }
}