// --- START OF FILE TreinamentExample.cs ---
namespace LLM.Models;

public class TreinamentExample
{
    // Construtor sem parâmetros necessário para deserialização JSON
    public TreinamentExample() { }

    public string Input { get; set; } = string.Empty; // Inicializar para evitar null
    public string Output { get; set; } = string.Empty; // Inicializar para evitar null

    // Construtor opcional para facilitar criação no código
    public TreinamentExample(string input, string output)
    {
        Input = input ?? throw new ArgumentNullException(nameof(input));
        Output = output ?? throw new ArgumentNullException(nameof(output));
    }
}
// --- END OF FILE TreinamentExample.cs ---