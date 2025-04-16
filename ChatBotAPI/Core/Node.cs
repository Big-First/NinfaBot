namespace ChatBotAPI.Core;

public class Node
{
    public int Token { get; set; }
    public List<Node> Children { get; set; } = new List<Node>();
    public List<List<int>> ResponseSequences { get; set; } = new List<List<int>>(); // Armazenar sequências completas de resposta
}