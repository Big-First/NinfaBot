namespace ChatBotAPI.Core;

public class TrainingExecutionState
{
    public bool ForceTraining { get; set; } = false; // Se o usuário digitou 'train'
    public bool WasModelLoaded { get; set; } = false; // Se o carregamento do arquivo foi bem-sucedido
    public bool ShouldRunTrainingBlock => ForceTraining || !WasModelLoaded; // Lógica combinada
}