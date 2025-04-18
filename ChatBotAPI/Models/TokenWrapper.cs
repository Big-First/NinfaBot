using System.Text.Json.Serialization;

namespace ChatBotAPI.Models;

public class TokenWrapper
{
    public TokenWrapper(){}
    public string version { get; set; }
    public object? normalizer { get; set; }
    public object? preTokenizer { get; set; }
    public List<AddedToken> addedTokens { get; set; }
    public object? truncation { get; set; }
    public object? padding { get; set; }
    public object? postProcessor { get; set; }
    public object? decoder { get; set; }
    public Model model { get; set; }

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