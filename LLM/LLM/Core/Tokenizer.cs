using Console = System.Console;
using Exception = System.Exception;
using System.Collections.Concurrent;
using SharpToken;
using System.Linq;

namespace LLM.Core;

public class Tokenizer
{
    private readonly GptEncoding _encoding;
    private readonly ConcurrentDictionary<string, int> _specialTokenIds = new(); // Cache

    // IDs padrão podem variar ligeiramente dependendo do encoding exato (gpt2, cl100k_base etc.)
    // Use os IDs retornados por SharpToken sempre que possível.
    public int PAD_ID { get; }
    public int EOS_ID { get; } // End of Sequence / End of Text
    public int START_ID { get; } // Opcional, pode não ser padrão em GPT
    public int UNK_ID { get; } // Opcional, SharpToken pode lidar com OOV

    // --- Opção 1: Usar um nome de encoding conhecido (ex: "gpt2", "cl100k_base") ---
    // Este construtor será usado com "gpt2"
    public Tokenizer(string encodingName = "gpt2") // Padrão para GPT-2
    {
        _encoding = GptEncoding.GetEncoding(encodingName);

        // Tenta obter IDs de tokens especiais comuns. O ID exato pode variar.
        // <|endoftext|> é comum para EOS em GPT-2/3
        EOS_ID = GetSpecialTokenId("<|endoftext|>");
        // GPT não costuma ter um PAD token formal no vocabulário base, mas podemos usar EOS ou outro ID.
        // Para consistência com o trainer e evitar erros, usaremos o EOS_ID como PAD_ID,
        // e o CrossEntropyLoss no trainer será configurado para ignorar este ID.
        PAD_ID = EOS_ID;
        // SharpToken lida com OOV (Out-Of-Vocabulary) dividindo em bytes/caracteres,
        // então um UNK dedicado pode não ser estritamente necessário como no BPE manual.
        UNK_ID = -1; // Indicar que não há UNK ID formal ou usar um ID se existir
        START_ID = -1; // Indicar que não há START ID formal ou usar um ID se existir

         Console.WriteLine($"Tokenizer '{encodingName}' inicializado. VocabSize: {_encoding.Count}, EOS_ID: {EOS_ID}, PAD_ID: {PAD_ID}");
    }

    // --- Opção 2: Carregar de arquivos vocab/merges (se você tiver um tokenizer BPE customizado) ---
    // public Tokenizer(string vocabPath, string mergesPath)
    // {
    //     // Nota: Você precisará garantir que os arquivos vocab.bpe e merges.txt
    //     //       estejam no formato esperado pelo SharpToken.
    //     //       SharpToken GetEncodingFromFilesAsync espera o formato usado por modelos GPT-2/3/Neo/J.
    //     //       O arquivo vocab.json tem pares token -> id
    //     //       O arquivo merges.txt tem pares de bytes/tokens a serem mesclados
    //     //       Você precisará de ambos se usar arquivos BPE customizados.
    //     //       SharpToken também precisa de um mapa de tokens especiais.
    //     var specialTokens = new Dictionary<string, int> { { "<|endoftext|>", 50256 } }; // Exemplo para GPT-2
    //     // Note: Este método pode ser assíncrono. Adapte conforme o uso.
    //     _encoding = GptEncoding.GetEncodingFromFilesAsync(vocabPath, mergesPath, specialTokens).GetAwaiter().GetResult();
    //
    //     // Tenta obter IDs de tokens especiais comuns.
    //     EOS_ID = GetSpecialTokenId("<|endoftext|>");
    //     PAD_ID = EOS_ID; // Usar EOS como PAD
    //     UNK_ID = -1;
    //     START_ID = -1;
    //
    //     Console.WriteLine($"Tokenizer carregado de arquivos inicializado. VocabSize: {_encoding.Count}, EOS_ID: {EOS_ID}, PAD_ID: {PAD_ID}");
    // }

    /// <summary>
    /// Codifica texto em uma lista de IDs de token.
    /// </summary>
    /// <param name="text">O texto a ser codificado.</param>
    /// <param name="addSpecialTokens">Se deve incluir tokens especiais reconhecidos pelo encoding (como <|endoftext|>).
    ///                                 Geralmente false para input do modelo, true para texto cru ou target de treino.</param>
    /// <returns>Lista de IDs de token.</returns>
    public List<int> Encode(string text, bool addSpecialTokens = true)
    {
        // SharpToken Encode já tem um parâmetro para lidar com tokens especiais permitidos.
        // A configuração 'addSpecialTokens' no SharpToken controla se os tokens especiais
        // encontrados no texto são *mantidos* ou quebrados em outros tokens.
        // Nosso parâmetro 'addSpecialTokens' aqui controla se SharpToken *deve procurá-los*.
        // Se setarmos permitted_special para um hashset vazio, SharpToken irá quebrar
        // tokens especiais mesmo que estejam no vocabulário, tratando-os como texto normal.
        // Isso é útil se você quer codificar texto bruto sem interpretação especial.
        // Se setarmos permitted_special para os tokens especiais, ele os manterá como tokens únicos.

        // Para input do modelo, geralmente não queremos que SharpToken adicione ou
        // interprete tokens especiais *que não estejam explicitamente no texto de entrada*.
        // Queremos apenas os IDs do texto fornecido.
        // A flag 'addSpecialTokens' do SharpToken refere-se mais a tokens especiais presentes no TEXTO.

        // Vamos usar o parâmetro bool 'addSpecialTokens' para controlar se
        // queremos incluir *nossos próprios* tokens especiais (que não é padrão GPT input)
        // ou apenas os tokens do texto. Para input do modelo (inferência/input do treino),
        // queremos apenas os IDs do texto. Para o target do treino, queremos adicionar EOS.

        // A implementação original do SharpToken Encode não adiciona EOS/BOS automaticamente
        // a menos que eles estejam presentes no texto E sejam tokens especiais permitidos.
        // O parâmetro `addSpecialTokens` no SharpToken `Encode` controla se os tokens especiais
        // (definidos no encoding) no texto de entrada são tratados como tokens ou como bytes.

        // Para nosso caso de uso (input do modelo), queremos apenas codificar o texto.
        // Adicionar EOS/BOS manualmente é feito na lógica de treinamento/inferência, não no tokenizer.Encode.
        // Portanto, o parâmetro `addSpecialTokens` aqui no *nosso* método `Encode`
        // não precisa modificar o comportamento subjacente do SharpToken `Encode`.
        // Ele apenas documenta que este método retorna IDs para o TEXTO fornecido.
        // A adição de EOS para o target do treino é feita *depois* de chamar Encode.

        // Implementação original do SharpToken Encode:
        // return _encoding.Encode(text);

        // Se quiséssemos que a flag `addSpecialTokens` *aqui* controlasse se SharpToken
        // deveria reconhecer tokens especiais NO TEXTO, faríamos:
        var permittedSpecialTokens = addSpecialTokens
            ? _encoding.SpecialTokens.Keys.ToHashSet() // Permite todos os tokens especiais conhecidos
            : new HashSet<string>(); // Não permite nenhum token especial (os trata como texto normal)

        return _encoding.Encode(text, permittedSpecialTokens);


        // Exemplo: Adicionar START/END (se definidos e desejado) - Isso não é padrão GPT
        // if (addSpecialTokens && START_ID != -1) ids.Insert(0, START_ID); // Não fazemos isso para input do modelo
        // if (addSpecialTokens && EOS_ID != -1) ids.Add(EOS_ID); // EOS geralmente é adicionado ao *target* no treino
    }

    public string Decode(List<int> ids)
    {
        // Remover tokens especiais *se necessário* antes de decodificar.
        // Na maioria dos casos, queremos decodificar tudo que o modelo gerou.
        // Ex: ids = ids.Where(id => id != PAD_ID && id != START_ID).ToList(); // Cuidado!
        // Decodificar o EOS_ID pode resultar na string "<|endoftext|>" dependendo do tokenizer.
        return _encoding.Decode(ids);
    }

    public int GetVocabSize()
    {
        return _encoding.Count;
    }

    public int GetEosTokenId()
    {
        // Certifique-se de que o EOS_ID foi encontrado. Se for -1, algo está errado.
        if (EOS_ID == -1)
        {
            Console.WriteLine("AVISO: EOS Token ID não encontrado para este encoding!");
            // Lançar exceção ou retornar um valor de fallback apropriado
            // throw new InvalidOperationException("EOS Token ID não encontrado.");
        }
        return EOS_ID;
    }

     public int GetPadTokenId()
    {
        // Assumimos que PAD_ID é configurado no construtor (usando EOS_ID por padrão).
         return PAD_ID;
    }

    /// <summary>
    /// Obtém o ID de um token especial, usando cache.
    /// Retorna -1 se o token não for encontrado no vocabulário.
    /// </summary>
    private int GetSpecialTokenId(string specialToken)
    {
        if (_specialTokenIds.TryGetValue(specialToken, out int cachedId))
        {
            return cachedId;
        }

        try
        {
            // Tenta obter o ID do token especial. SharpToken pode lançar erro se não for um token especial conhecido.
             // A forma mais segura é usar Encode com apenas o token especial e verificar se retorna 1 ID.
             var encoded = _encoding.Encode(specialToken, new HashSet<string> { specialToken }); // Permite apenas ESTE token especial

             if (encoded.Count == 1)
             {
                 int id = encoded[0];
                 _specialTokenIds.TryAdd(specialToken, id);
                 return id;
             } else if (encoded.Count > 1)
             {
                 Console.WriteLine($"AVISO: Token especial '{specialToken}' codifica para múltiplos IDs ({encoded.Count}). Pode não ser um token especial único no vocabulário.");
                  _specialTokenIds.TryAdd(specialToken, -1); // Marca como não encontrado como token único especial
                  return -1; // Não é um token especial único
             }
             else // encoded.Count == 0 (deve ser impossível para string não vazia)
             {
                  Console.WriteLine($"AVISO: Token especial '{specialToken}' não pôde ser codificado.");
                  _specialTokenIds.TryAdd(specialToken, -1); // Marca como não encontrado
                  return -1;
             }

        }
        catch (Exception ex) // Captura outros erros potenciais de encoding
        {
             Console.WriteLine($"ERRO ao obter ID para token especial '{specialToken}': {ex.Message}");
             _specialTokenIds.TryAdd(specialToken, -1); // Marca como não encontrado
             return -1;
        }
    }
}
// --- END OF FILE Tokenizer.cs ---