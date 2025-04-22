// --- START OF FILE PromptRequest.cs ---
namespace LLM.Models;

public class PromptRequest
{
    public string Input { get; set; } = string.Empty;

    // Parâmetros de Geração (com valores padrão)
    public int MaxNewTokens { get; set; } = 50;
    public double Temperature { get; set; } = 0.7;
    public int TopK { get; set; } = 0; // 0 significa desabilitado
    public double TopP { get; set; } = 0.9; // 0 significa desabilitado
}
// --- END OF FILE PromptRequest.cs ---