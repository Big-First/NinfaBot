namespace ChatBotAPI.Core;

public class Node
{
    // O ID do token que este nó representa na sequência.
    public int Token { get; set; }

    // Filhos deste nó, mapeados pelo ID do token filho para o nó filho correspondente.
    // Substitui Left/Right por uma estrutura de dicionário (Trie-like).
    public Dictionary<int, Node> Filhos { get; set; }

    // Lista de IDs de token que foram observados seguindo a sequência que termina neste nó.
    public List<int> NextTokens { get; set; }

    // Construtor sem parâmetros para desserialização e inicialização
    public Node()
    {
        // Inicializa as coleções para evitar NullReferenceException
        Filhos = new Dictionary<int, Node>();
        NextTokens = new List<int>();
    }
}