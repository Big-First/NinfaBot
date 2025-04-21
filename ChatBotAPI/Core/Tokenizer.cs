// Tokenizer.cs

using System;
using System.Collections.Generic;
using System.Linq;
using SharpToken;

namespace ChatBotAPI.Core
{
    public partial class Tokenizer // Mantém partial se houver outra parte
    {
        private readonly GptEncoding _gptEncoding;
        private readonly int maxSequenceLength; // Max len para TRUNCAMENTO, não padding aqui

        // --- IDs ---
        public int PadTokenId { get; private set; }
        public int UnkTokenId { get; private set; }
        public int EosTokenId { get; private set; }
        public int VocabSize { get; private set; }

        private const int StandardGpt2VocabSize = 50257;
        private const int StandardGpt2EosPadId = 50256;

        // Construtor (como antes)
        public Tokenizer(int maxSequenceLength)
        {
             this.maxSequenceLength = maxSequenceLength; // Guarda o max len para truncamento opcional
             Console.WriteLine($"Initializing Tokenizer using SharpToken for GPT-2 (r50k_base) encoding. MaxSeqLen for truncation/padding = {maxSequenceLength}");
             try {
                 const string encodingName = "r50k_base";
                 _gptEncoding = GptEncoding.GetEncoding(encodingName);
                 if (_gptEncoding == null) throw new InvalidOperationException(/*...*/);

                 this.PadTokenId = StandardGpt2EosPadId; // 50256
                 this.EosTokenId = StandardGpt2EosPadId; // 50256
                 this.UnkTokenId = -1;
                 this.VocabSize = StandardGpt2VocabSize; // 50257

                 Console.WriteLine("Tokenizer initialized using SharpToken.");
                 Console.WriteLine("--> Using Standard GPT-2 Values for Model Config:");
                 Console.WriteLine($"    Vocab Size: {this.VocabSize}");
                 Console.WriteLine($"    Pad Token ID: {this.PadTokenId} ('<|endoftext|>')");
                 Console.WriteLine($"    EOS Token ID: {this.EosTokenId} ('<|endoftext|>')");
                 Console.WriteLine($"    UNK Token ID: {this.UnkTokenId} (N/A)");
             } catch (Exception ex) { /*...*/ throw; }
        }

        public int ActualVocabSize => this.VocabSize;

        // ***** MÉTODO Tokenize MODIFICADO *****
        public int[] Tokenize(string text, HashSet<string>? allowedSpecial = null, bool applyPaddingTruncation = true)
        {
            if (_gptEncoding == null) throw new InvalidOperationException("SharpToken encoding not initialized.");
            try
            {
                // Usa o allowedSpecial passado ou um default se nulo
                var effectiveAllowedSpecial = allowedSpecial ?? new HashSet<string> { "<|endoftext|>" };

                // Codifica o texto
                List<int> tokens = _gptEncoding.Encode(text, allowedSpecial: effectiveAllowedSpecial);

                // ***** APLICA PADDING/TRUNCAMENTO APENAS SE SOLICITADO *****
                if (applyPaddingTruncation)
                {
                    int currentLength = tokens.Count;
                    if (currentLength > maxSequenceLength)
                    {
                        // Trunca para o tamanho máximo
                        tokens = tokens.GetRange(0, maxSequenceLength);
                         // Console.WriteLine($"Warning: Input text truncated to {maxSequenceLength} tokens.");
                    }
                    else if (currentLength < maxSequenceLength)
                    {
                        // Adiciona Padding (ID 50256) até o tamanho máximo
                        tokens.AddRange(Enumerable.Repeat(this.PadTokenId, maxSequenceLength - currentLength));
                    }
                }
                // Se applyPaddingTruncation for false, retorna apenas os tokens reais da codificação

                return tokens.ToArray();
            }
            catch (Exception ex) { Console.Error.WriteLine($"SharpToken Tokenize Error: {ex}"); throw; }
        }
         // ***** FIM MÉTODO Tokenize MODIFICADO *****


        // Método Detokenize (como antes)
        public string Detokenize(int[] tokens)
        {
             if (_gptEncoding == null) throw new InvalidOperationException("SharpToken encoding not initialized.");
             try
             {
                 List<int> idsToDecode = tokens.Where(t => t != this.PadTokenId).ToList();
                 if (idsToDecode.Count == 0) return "";
                 string decodedText = _gptEncoding.Decode(idsToDecode);
                 return decodedText.Trim();
             }
             catch (Exception ex) { Console.Error.WriteLine($"SharpToken Detokenize Error: {ex}"); return "[Detokenization Error]"; }
        }

        // Método GetMaxSequenceLength (como antes)
        public int GetMaxSequenceLength()
        {
            return this.maxSequenceLength;
        }
    }
}