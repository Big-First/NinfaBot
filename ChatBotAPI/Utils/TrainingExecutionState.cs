namespace ChatBotAPI.Utils
{
    /// <summary>
    /// Gerencia o estado da execução do treinamento.
    /// </summary>
    public class TrainingExecutionState
    {
        /// <summary>
        /// Indica se o treinamento deve ser forçado, mesmo se um modelo existir.
        /// </summary>
        public bool ForceTraining { get; set; } = false;

        /// <summary>
        /// Indica se um modelo foi carregado com sucesso ao iniciar.
        /// </summary>
        public bool WasModelLoaded { get; set; } = false;

        /// <summary>
        /// Determina se o bloco de treinamento deve ser executado.
        /// </summary>
        /// <remarks>
        /// Executa se ForceTraining for true OU se ForceTraining for false E nenhum modelo foi carregado.
        /// </remarks>
        public bool ShouldRunTrainingBlock => ForceTraining || !WasModelLoaded;

        public TrainingExecutionState()
        {
            Console.WriteLine("TrainingExecutionState initialized.");
        }
    }
}