namespace ConsoleApp;

public class TokenData
{
    public Model model { get; set; }
    public object normalizer { get; set; }
    public object preTokenizer { get; set; }
    public List<AddedToken> addedTokens { get; set; }
    public object truncation { get; set; }
    public object padding { get; set; }
    public object postProcessor { get; set; }
    public object decoder { get; set; }
    public Dictionary<string, object> specialTokens { get; set; }
}