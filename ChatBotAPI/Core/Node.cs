namespace ChatBotAPI.Core;

public class Node
{
    public int Token { get; set; }
    public Node Left { get; set; }
    public Node Right { get; set; }
    public List<int> NextTokens { get; set; } = new List<int>();
}