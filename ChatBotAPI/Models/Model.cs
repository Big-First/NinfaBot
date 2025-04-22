namespace ChatBotAPI.Models;

/// <summary>
/// Represents a language model configuration with its type and vocabulary.
/// </summary>
public class Model
{
    /// <summary>
    /// Gets or sets the type of the model.
    /// </summary>
    public string type { get; set; }

    /// <summary>
    /// Gets or sets the vocabulary dictionary mapping tokens to their integer IDs.
    /// </summary>
    public Dictionary<string, int> vocab { get; set; }
}