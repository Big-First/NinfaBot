using System.Collections.Concurrent;
using SharpToken;

namespace LLM.Core;

public class Tokenizer
{
    private readonly GptEncoding _encoding;
    private readonly ConcurrentDictionary<string, int> _specialTokenIds = new();

    public int PAD_ID { get; }
    public int EOS_ID { get; }
    public int START_ID { get; }
    public int UNK_ID { get; }

    public Tokenizer(string encodingName = "gpt2")
    {
        try
        {
            _encoding = GptEncoding.GetEncoding(encodingName);

            // --- COMENTAR TEMPORARIAMENTE AS CHAMADAS PARA GetSpecialTokenId ---
            // EOS_ID = GetSpecialTokenId("<|endoftext|>"); // COMENTADO
            // PAD_ID = EOS_ID; // COMENTADO
            // START_ID = GetSpecialTokenId("<|startoftext|>"); // COMENTADO
            // UNK_ID = -1; // Manter UNK_ID como -1 (não depende de GetSpecialTokenId)

            // INICIALIZAR COM VALORES DEFAULT QUANDO COMENTADO PARA COMPILAR
            EOS_ID = -1; // Valor temporário
            PAD_ID = -1; // Valor temporário
            START_ID = -1; // Valor temporário
            // UNK_ID já está -1

            // --- FIM DOS COMENTÁRIOS TEMPORÁRIOS ---


            System.Console.WriteLine("----------------------------------------------------");
            System.Console.WriteLine($"Tokenizer '{encodingName}' inicializado.");
            // --- TESTE DRÁSTICO NESTA LINHA ESPECÍFICA ---
            System.Console.WriteLine("  VocabSize: [Valor Removido Para Teste]"); // Apenas uma string literal
            // --- FIM DO TESTE DRÁSTICO ---
            System.Console.WriteLine("  EOS_ID: {0} ('<|endoftext|>')", EOS_ID);
            System.Console.WriteLine("  PAD_ID: {0} (usando EOS_ID)", PAD_ID);
            System.Console.WriteLine("  START_ID: {0}", (START_ID != -1 ? START_ID.ToString() : "Não encontrado ('<|startoftext|>')"));
            System.Console.WriteLine("----------------------------------------------------");


             if (EOS_ID == -1)
             {
                  Console.WriteLine("ERRO: EOS Token ID (<|endoftext|>) não encontrado para este encoding!");
             }

        }
        // Captura System.Exception (graças ao using System;)
        catch (System.Exception ex)
        {
            System.Console.WriteLine($"ERRO fatal ao inicializar Tokenizer com encoding '{encodingName}': {ex.Message}");
            System.Console.WriteLine(ex.StackTrace);
            // Lança System.InvalidOperationException (graças ao using System;)
            throw new System.InvalidOperationException($"Falha ao inicializar Tokenizer: {ex.Message}", ex);
        }
    }

    // ... (Restante da classe Tokenizer - métodos Encode, Decode, GetVocabSize, GetEosTokenId, GetPadTokenId, GetSpecialTokenId) ...

     /// <summary>
     /// Codifica texto em uma lista de IDs de token.
     /// </summary>
     /// <param name="text">O texto a ser codificado.</param>
     /// <param name="allowSpecialTokensInText">Controla se tokens especiais ENCONTRADOS NO TEXTO de entrada
     ///                                       devem ser tratados como tokens únicos ou quebrados em bytes/subpalavras.</param>
     /// <returns>Lista de IDs de token.</returns>
     public List<int> Encode(string text, bool allowSpecialTokensInText = true)
     {
         HashSet<string> allowedSpecial;
         if (allowSpecialTokensInText)
         {
             // CORREÇÃO AQUI: Se _encoding.SpecialTokens não existe ou não é público,
             // precisamos encontrar outra forma de obter a lista de tokens especiais.
             // Uma alternativa é criar a HashSet manualmente se soubermos quais tokens especiais usar.
             // Para GPT-2, o principal token especial é "<|endoftext|>".
             // SharpToken pode reconhecer outros tokens especiais definidos no encoding (como em cl100k_base).
             // Para a versão 2.0.3, pode ser que não haja uma propriedade pública fácil.
             // Vamos tentar criar manualmente os tokens especiais conhecidos para GPT-2:
             allowedSpecial = new HashSet<string> { "<|endoftext|>" };
             // Se você precisa de outros tokens especiais (como em cl100k_base, tipo <|im_start|>), adicione-os aqui.
             // Ex para cl100k_base: new HashSet<string> { "<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|endofprompt|>" };

             // Nota: Se allowSpecialTokensInText for false, queremos uma HashSet vazia,
             // que já está sendo tratada corretamente no bloco 'else'.

         }
         else
         {
             // Não permite que SharpToken reconheça NENHUM token especial no texto de entrada.
             // Trata todos os tokens especiais como texto comum.
             allowedSpecial = new HashSet<string>();
         }

         // A chamada para _encoding.Encode(text, allowedSpecial) deve funcionar,
         // pois a sobrecarga com HashSet<string> parece existir.
         return _encoding.Encode(text, allowedSpecial);
     }

     /// <summary>
     /// Decodifica uma lista de IDs de token de volta para texto.
     /// </summary>
     /// <param name="ids">A lista de IDs de token.</param>
     /// <returns>O texto decodificado.</returns>
     public string Decode(List<int> ids)
     {
         return _encoding.Decode(ids);
     }

     /// <summary>
     /// Obtém o tamanho do vocabulário do tokenizer.
     /// </summary>
     public int GetVocabSize()
     {
         return 50257;
     }

     /// <summary>
     /// Obtém o ID do token de fim de sequência (EOS).
     /// </summary>
     public int GetEosTokenId()
     {
         if (EOS_ID == -1)
         {
              System.Console.WriteLine("AVISO: GetEosTokenId chamado, mas EOS Token ID não foi encontrado durante a inicialização.");
         }
         return EOS_ID;
     }

     /// <summary>
     /// Obtém o ID do token de padding (PAD).
     /// </summary>
      public int GetPadTokenId()
     {
          return PAD_ID;
     }

     /// <summary>
     /// Obtém o ID de um token especial específico, usando cache.
     /// Retorna -1 se o token não for encontrado no vocabulário ou não for reconhecido como token especial único.
     /// </summary>
     private int GetSpecialTokenId(string specialToken)
     {
         if (_specialTokenIds.TryGetValue(specialToken, out int cachedId))
         {
             return cachedId;
         }

         try
         {
              var encoded = _encoding.Encode(specialToken, new HashSet<string> { specialToken });

              if (encoded.Count == 1)
              {
                  int id = encoded[0];
                  _specialTokenIds.TryAdd(specialToken, id);
                  return id;
              }
              else
              {
                  System.Console.WriteLine($"AVISO: Token especial '{specialToken}' não encontrado como token único no vocabulário do encoding '{_encoding}'. Codificado para {encoded.Count} IDs.");
                   _specialTokenIds.TryAdd(specialToken, -1);
                   return -1;
              }
         }
         // Captura System.Exception (graças ao using System;)
         catch (System.Exception ex)
         {
              System.Console.WriteLine($"ERRO ao obter ID para token especial '{specialToken}': {ex.Message}");
              _specialTokenIds.TryAdd(specialToken, -1);
              return -1;
         }
     }
}
// --- END OF FILE Tokenizer.cs ---