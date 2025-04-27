namespace AI.Core
{
    /// <summary>
    /// Modelo para a requisição de resumo. Pode conter texto direto ou um caminho de arquivo.
    /// </summary>
    public class SummarizeRequest
    {
        /// <summary>
        /// O texto a ser resumido. Use esta propriedade OU FilePath.
        /// </summary>
        public string? Text { get; set; }

        /// <summary>
        /// O caminho absoluto ou relativo no servidor para o arquivo (.pdf, .docx, .txt) a ser lido e resumido.
        /// Use esta propriedade OU Text.
        /// </summary>
        /// <remarks>
        /// O uso desta propriedade implica que o arquivo deve estar acessível pelo servidor onde a API está rodando.
        /// Em ambientes de produção, caminhos de arquivo fornecidos pelo usuário podem representar riscos de segurança.
        /// Considere validar ou restringir os caminhos permitidos.
        /// </remarks>
        public string? FilePath { get; set; }

        /// <summary>
        /// O número máximo aproximado de tokens desejado para o resumo.
        /// Valor padrão se não especificado.
        /// </summary>
        public int MaxTokens { get; set; } = 200; // Valor padrão razoável
    }
}