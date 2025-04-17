namespace ConsoleApp;

public class Model
{
    public string type { get; set; }
    public Dictionary<string, int> vocab { get; set; }
    public List<string> merges { get; set; }

    public Model(string type, Dictionary<string, int> vocab, List<string> merges)
    {
        this.type = type;
        this.vocab = vocab;
        this.merges = merges;
    }
}