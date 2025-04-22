namespace ChatBotAPI.Models;

/// <summary>
/// Represents an additional token that can be added to the tokenizer's vocabulary.
/// </summary>
public class AddedToken
{
    /// <summary>
    /// Initializes a new instance of the AddedToken class.
    /// </summary>
    public AddedToken(){}

    /// <summary>
    /// Gets or sets the content of the token.
    /// </summary>
    public string content { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether the token should be treated as a single word.
    /// </summary>
    public bool single_word { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether to strip whitespace from the left of the token.
    /// </summary>
    public bool lstrip { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether to strip whitespace from the right of the token.
    /// </summary>
    public bool rstrip { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether the token should be normalized.
    /// </summary>
    public bool normalized { get; set; }

    /// <summary>
    /// Gets or sets the special token identifier, if this token is a special token.
    /// </summary>
    public int? special { get; set; }

    /// <summary>
    /// Initializes a new instance of the AddedToken class with specified parameters.
    /// </summary>
    /// <param name="content">The content of the token.</param>
    /// <param name="singleWord">Whether the token should be treated as a single word.</param>
    /// <param name="lstrip">Whether to strip whitespace from the left of the token.</param>
    /// <param name="rstrip">Whether to strip whitespace from the right of the token.</param>
    /// <param name="normalized">Whether the token should be normalized.</param>
    /// <param name="special">The special token identifier, if this token is a special token.</param>
    public AddedToken(string content, bool singleWord, bool lstrip, bool rstrip, bool normalized, int? special)
    {
        this.content = content;
        single_word = singleWord;
        this.lstrip = lstrip;
        this.rstrip = rstrip;
        this.normalized = normalized;
        this.special = special;
    }
}