using System.Text.Json.Serialization;

namespace ChatBotAPI.Models;

/// <summary>
/// Wrapper class that contains tokenizer configuration and settings.
/// </summary>
public class TokenWrapper
{
    /// <summary>
    /// Initializes a new instance of the TokenWrapper class.
    /// </summary>
    public TokenWrapper(){}

    /// <summary>
    /// Gets or sets the version of the tokenizer.
    /// </summary>
    public string version { get; set; }

    /// <summary>
    /// Gets or sets the normalizer configuration.
    /// </summary>
    public object? normalizer { get; set; }

    /// <summary>
    /// Gets or sets the pre-tokenizer configuration.
    /// </summary>
    public object? preTokenizer { get; set; }

    /// <summary>
    /// Gets or sets the list of additional tokens.
    /// </summary>
    public List<AddedToken> addedTokens { get; set; }

    /// <summary>
    /// Gets or sets the truncation configuration.
    /// </summary>
    public object? truncation { get; set; }

    /// <summary>
    /// Gets or sets the padding configuration.
    /// </summary>
    public object? padding { get; set; }

    /// <summary>
    /// Gets or sets the post-processor configuration.
    /// </summary>
    public object? postProcessor { get; set; }

    /// <summary>
    /// Gets or sets the decoder configuration.
    /// </summary>
    public object? decoder { get; set; }

    /// <summary>
    /// Gets or sets the model configuration.
    /// </summary>
    public Model model { get; set; }

    /// <summary>
    /// Initializes a new instance of the TokenWrapper class with specified parameters.
    /// </summary>
    /// <param name="version">The version of the tokenizer.</param>
    /// <param name="normalizer">The normalizer configuration.</param>
    /// <param name="preTokenizer">The pre-tokenizer configuration.</param>
    /// <param name="addedTokens">The list of additional tokens.</param>
    /// <param name="truncation">The truncation configuration.</param>
    /// <param name="padding">The padding configuration.</param>
    /// <param name="postProcessor">The post-processor configuration.</param>
    /// <param name="decoder">The decoder configuration.</param>
    /// <param name="model">The model configuration.</param>
    public TokenWrapper(string version, object normalizer, object preTokenizer, List<AddedToken> addedTokens, object truncation, object padding, object postProcessor, object decoder, Model model)
    {
        this.version = version;
        this.normalizer = normalizer;
        this.preTokenizer = preTokenizer;
        this.addedTokens = addedTokens;
        this.truncation = truncation;
        this.padding = padding;
        this.postProcessor = postProcessor;
        this.decoder = decoder;
        this.model = model;
    }
}