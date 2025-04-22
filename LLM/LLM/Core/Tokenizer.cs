using System.Collections.Concurrent;
using SharpToken;

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
    public Tokenizer(string encodingName = "gpt2") // Padrão para GPT-2
    {
        _encoding = GptEncoding.GetEncoding(encodingName);

        // Tenta obter IDs de tokens especiais comuns. O ID exato pode variar.
        // <|endoftext|> é comum para EOS em GPT-2/3
        EOS_ID = GetSpecialTokenId("<|endoftext|>");
        // GPT não costuma ter um PAD token formal no vocabulário base, mas podemos usar EOS ou outro ID.
        // Se precisar de um PAD distinto, você teria que adicioná-lo ao SharpToken (mais avançado).
        // Por simplicidade, podemos usar EOS como PAD se o modelo for treinado para ignorá-lo,
        // ou escolher um ID não utilizado (arriscado), ou usar o próprio EOS_ID. Vamos usar EOS por enquanto.
        PAD_ID = EOS_ID;
        // SharpToken lida com OOV (Out-Of-Vocabulary) dividindo em bytes/caracteres,
        // então um UNK dedicado pode não ser estritamente necessário como no BPE manual.
        UNK_ID = -1; // Indicar que não há UNK ID formal ou usar um ID se existir
        START_ID = -1; // Indicar que não há START ID formal ou usar um ID se existir
    }

    // --- Opção 2: Carregar de arquivos vocab/merges (se você tiver um tokenizer BPE customizado) ---
    // public Tokenizer(string vocabPath, string mergesPath)
    // {
    //     // Nota: Você precisará garantir que os arquivos vocab.bpe e merges.txt
    //     //       estejam no formato esperado pelo SharpToken.
    //     var specialTokens = new Dictionary<string, int> { { "<|endoftext|>", 50256 } }; // Exemplo
    //     _encoding = GptEncoding.GetEncodingFromFilesAsync(vocabPath, mergesPath, specialTokens).GetAwaiter().GetResult();
    //
    //     EOS_ID = _encoding.EncodeSingleToken("<|endoftext|>");
    //     PAD_ID = EOS_ID; // Ou outra estratégia de padding
    //     // ... configurar outros IDs especiais se existirem no seu vocabulário
    // }

    public List<int> Encode(string text, bool addSpecialTokens = true)
    {
        // Nota: SharpToken pode ter sua própria lógica para tokens especiais permitidos.
        // Ajuste conforme necessário. A adição manual de START/END pode ser feita aqui se desejado.
        var ids = _encoding.Encode(text);

        // Exemplo: Adicionar START/END (se definidos e desejado) - Isso não é padrão GPT
        // if (addSpecialTokens && START_ID != -1) ids.Insert(0, START_ID);
        // if (addSpecialTokens && EOS_ID != -1) ids.Add(EOS_ID); // EOS geralmente é adicionado ao *target* no treino

        return ids;
    }

    public string Decode(List<int> ids)
    {
        // Remover tokens especiais antes de decodificar, se necessário
        // Ex: ids = ids.Where(id => id != PAD_ID && id != START_ID).ToList();
        // Cuidado ao remover EOS se ele for parte legítima do texto gerado.
        return _encoding.Decode(ids);
    }

    public int GetVocabSize()
    {
        return _encoding.Count;
    }

    public int GetEosTokenId()
    {
        return EOS_ID;
    }

    public int GetPadTokenId()
    {
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
            // EncodeSingleToken pode lançar exceção se o token não for encontrado diretamente
            int id = _encoding.EncodeSingleToken(specialToken);
            _specialTokenIds.TryAdd(specialToken, id);
            return id;
        }
        catch (KeyNotFoundException)
        {
            // Tenta encodar como texto normal para ver se ele é composto por outros tokens
            var encodedList = _encoding.Encode(specialToken);
            if (encodedList.Count == 1) // Só consideramos se for um único token após encoding
            {
                _specialTokenIds.TryAdd(specialToken, encodedList[0]);
                return encodedList[0];
            }
            _specialTokenIds.TryAdd(specialToken, -1); // Marca como não encontrado
            Console.WriteLine($"AVISO: Token especial '{specialToken}' não encontrado no vocabulário.");
            return -1;
        }
        catch (Exception ex) // Outras exceções inesperadas
        {
             Console.WriteLine($"ERRO ao obter ID para token especial '{specialToken}': {ex.Message}");
             _specialTokenIds.TryAdd(specialToken, -1); // Marca como não encontrado
             return -1;
        }
    }
}
// --- END OF FILE Tokenizer.cs ---