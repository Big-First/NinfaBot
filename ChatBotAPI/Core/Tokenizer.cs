// Tokenizer.cs - Permitindo <|endoftext|> no Encode

using System;
using System.Collections.Generic;
using System.Linq;
using SharpToken; // Garanta o using

namespace ChatBotAPI.Core
{
    public partial class Tokenizer
    {
        private readonly GptEncoding _gptEncoding;

        public int PadTokenId { get; private set; }
        public int UnkTokenId { get; private set; }
        public int EosTokenId { get; private set; }
        public int VocabSize { get; private set; }

        private const int StandardGpt2VocabSize = 50257;
        private const int StandardGpt2EosPadId = 50256;
        private const string StandardGpt2EosToken = "<|endoftext|>"; // Guarda a string

        // --- NOVO: Conjunto de tokens especiais a permitir no Encode ---
        private readonly HashSet<string> _allowedSpecialTokens;

        private readonly int maxSequenceLength;

        public Tokenizer(int maxSequenceLength)
        {
            this.maxSequenceLength = maxSequenceLength;
            Console.WriteLine($"Initializing Tokenizer using SharpToken for GPT-2 (r50k_base) encoding.");

            try
            {
                const string encodingName = "r50k_base";
                _gptEncoding = GptEncoding.GetEncoding(encodingName);
                if (_gptEncoding == null) throw new InvalidOperationException(/*...*/);
                Console.WriteLine($"Successfully obtained '{encodingName}' encoding from SharpToken."); // Usei ??

                // Define IDs e VocabSize
                this.PadTokenId = StandardGpt2EosPadId;
                this.EosTokenId = StandardGpt2EosPadId;
                this.UnkTokenId = -1;
                this.VocabSize = StandardGpt2VocabSize;

                // *** INICIALIZA O CONJUNTO DE TOKENS PERMITIDOS ***
                // Começa permitindo o token EOS/PAD padrão que estamos usando
                this._allowedSpecialTokens = new HashSet<string> { StandardGpt2EosToken };
                // Você poderia adicionar outros tokens especiais aqui se os usasse
                // Ex: _allowedSpecialTokens.Add("<|im_start|>");

                Console.WriteLine($"Tokenizer initialized using SharpToken.");
                Console.WriteLine($"--> Using Standard GPT-2 Values for Model Config:");
                Console.WriteLine($"    Vocab Size: {this.VocabSize}");
                Console.WriteLine($"    Pad Token ID: {this.PadTokenId} ('{StandardGpt2EosToken}')"); // Mostra string
                Console.WriteLine($"    EOS Token ID: {this.EosTokenId} ('{StandardGpt2EosToken}')"); // Mostra string
                Console.WriteLine($"    UNK Token ID: {this.UnkTokenId} (N/A)");
                Console.WriteLine($"    Allowed Special Tokens for Encode: [{string.Join(", ", _allowedSpecialTokens)}]"); // Log dos permitidos

            }
            catch (Exception ex) { /*...*/ throw; }
        }

        // Property ActualVocabSize
        public int ActualVocabSize => this.VocabSize;

        // --- Método Tokenize MODIFICADO ---
        public int[] Tokenize(string text)
        {
            if (_gptEncoding == null) throw new InvalidOperationException("SharpToken encoding not initialized.");
            try
            {
                // *** CHAMA A SOBRECARGA DE Encode passando allowedSpecial ***
                List<int> tokens = _gptEncoding.Encode(text, allowedSpecial: _allowedSpecialTokens);

                // Aplica Truncamento e Padding MANUALMENTE
                int currentLength = tokens.Count;
                if (currentLength > maxSequenceLength) { tokens = tokens.GetRange(0, maxSequenceLength); }
                else if (currentLength < maxSequenceLength) { tokens.AddRange(Enumerable.Repeat(this.PadTokenId, maxSequenceLength - currentLength)); }
                return tokens.ToArray();
            }
            catch (ArgumentException argEx) when (argEx.Message.Contains("Disallowed special token"))
            {
                 // Log mais específico se o erro persistir (não deveria)
                 Console.Error.WriteLine($"SharpToken Tokenize Error: Still encountered a disallowed token even after allowing some. Input text (start): '{text.Substring(0, Math.Min(50, text.Length))}'");
                 Console.Error.WriteLine($"Full Error: {argEx}");
                 throw; // Re-lança
            }
            catch (Exception ex) { Console.Error.WriteLine($"SharpToken Tokenize Error: {ex}"); throw; }
        }

        // --- Método Detokenize (Inalterado) ---
        public string Detokenize(int[] tokens)
        {
             if (_gptEncoding == null) throw new InvalidOperationException("SharpToken encoding not initialized.");
              try {
                  // SharpToken Decode lida com <|endoftext|> por padrão
                  List<int> idsToDecode = tokens.Where(t => t != this.PadTokenId).ToList(); // Remove padding antes
                  if (idsToDecode.Count == 0) return "";
                  // O Decode padrão deve remover o <|endoftext|> se ele estiver na lista idsToDecode
                  string decodedText = _gptEncoding.Decode(idsToDecode);
                  return decodedText.Trim();
              } catch (Exception ex) { /*...*/ return "[Detokenization Error]"; }
        }

        // Método GetMaxSequenceLength
        public int GetMaxSequenceLength()
        {
            return this.maxSequenceLength;
        }
    }
}