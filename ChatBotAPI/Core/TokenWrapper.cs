using System.Text.Json.Serialization;

namespace ChatBotAPI.Core;

public class TokenWrapper
{
    public string Version { get; set; }
    public Model Model { get; set; }
}