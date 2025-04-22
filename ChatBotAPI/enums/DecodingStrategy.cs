namespace ChatBotAPI.enums;

/// <summary>
/// Defines the different strategies for decoding text generation in the chatbot.
/// These strategies determine how the model selects the next token during text generation.
/// </summary>
public enum DecodingStrategy
{
    /// <summary>
    /// Uses temperature, top-k, and top-p sampling to generate diverse and creative responses.
    /// This is the default strategy that provides a good balance between coherence and creativity.
    /// </summary>
    Sampling, // Default: Usa Temperatura, Top-K, Top-P

    /// <summary>
    /// Always selects the token with the highest probability at each step.
    /// This strategy is deterministic and produces the most likely output sequence.
    /// </summary>
    Greedy,

    /// <summary>
    /// Maintains multiple candidate sequences and expands them to find the best overall sequence.
    /// This is a placeholder for future implementation as it requires complex logic.
    /// </summary>
    BeamSearch // Placeholder - implementação complexa
}