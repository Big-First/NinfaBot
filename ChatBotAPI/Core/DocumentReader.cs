using System.Text;
using NPOI.XWPF.UserModel;
using UglyToad.PdfPig;

namespace AI.Core
{
    /// <summary>
    /// Serviço para ler conteúdo de texto de diferentes formatos de documento.
    /// Suporta .pdf e .docx.
    /// </summary>
    public class DocumentReader
    {
        /// <summary>
        /// Lê o texto de um arquivo especificado.
        /// </summary>
        /// <param name="filePath">O caminho completo para o arquivo.</param>
        /// <returns>O conteúdo de texto do arquivo.</returns>
        /// <exception cref="FileNotFoundException">Lançada se o arquivo não for encontrado.</exception>
        /// <exception cref="NotSupportedException">Lançada se o formato do arquivo não for suportado.</exception>
        /// <exception cref="Exception">Lançada para outros erros de leitura.</exception>
        public string ReadDocument(string filePath)
        {
            if (!File.Exists(filePath))
            {
                Console.Error.WriteLine($"DocumentReader Error: File not found at {filePath}");
                throw new FileNotFoundException($"Arquivo não encontrado: {filePath}");
            }

            string extension = Path.GetExtension(filePath).ToLowerInvariant();

            try
            {
                switch (extension)
                {
                    case ".pdf":
                        return ReadPdf(filePath);
                    case ".docx":
                        return ReadDocx(filePath);
                    // Adicione outros formatos aqui se necessário (ex: .txt)
                    case ".txt":
                        return File.ReadAllText(filePath);
                    default:
                        Console.Error.WriteLine($"DocumentReader Error: Unsupported file format: {extension} for {filePath}");
                        throw new NotSupportedException($"Formato de arquivo não suportado: {extension}");
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"DocumentReader Error reading {filePath}: {ex.Message}");
                throw new Exception($"Erro ao ler o arquivo {Path.GetFileName(filePath)}: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Lê o texto de um arquivo PDF.
        /// </summary>
        private string ReadPdf(string filePath)
        {
            StringBuilder text = new StringBuilder();
            using (var document = PdfDocument.Open(filePath))
            {
                foreach (var page in document.GetPages())
                {
                    text.AppendLine(page.Text);
                }
            }
            Console.WriteLine($"Successfully read PDF file: {filePath}");
            return text.ToString();
        }

        /// <summary>
        /// Lê o texto de um arquivo DOCX.
        /// </summary>
        private string ReadDocx(string filePath)
        {
            StringBuilder text = new StringBuilder();
            using (FileStream fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read))
            {
                XWPFDocument doc = new XWPFDocument(fileStream);
                foreach (var paragraph in doc.Paragraphs)
                {
                    text.AppendLine(paragraph.Text);
                }
            }
             Console.WriteLine($"Successfully read DOCX file: {filePath}");
            return text.ToString();
        }
    }
}