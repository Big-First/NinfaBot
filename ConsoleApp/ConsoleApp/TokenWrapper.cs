namespace ConsoleApp;

public class TokenWrapper
{
    public TokenWrapper(){}
    public string version { get; set; }
    public Model model { get; set; }

    public TokenWrapper(string version, Model model)
    {
        this.version = version;
        this.model = model;
    }
}