using System.Text.Json;
using System.Text.Json.Serialization;

namespace ChatBotAPI.Core;

public class Token
{
    [JsonPropertyName("id")]
    public int Id { get; set; }

    [JsonExtensionData]
    public Dictionary<string, JsonElement> ExtraData { get; set; }
}