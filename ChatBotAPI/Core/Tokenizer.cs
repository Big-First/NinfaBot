using SharpToken;

namespace AI.Core
{
    /// <summary>
    /// Provides tokenization functionality for text processing using GPT-2 encoding.
    /// This class handles text tokenization, detokenization, and manages token-related operations
    /// using the SharpToken library for GPT-2 (r50k_base) encoding.
    /// </summary>
    public partial class Tokenizer // Mantém partial se houver outra parte
    {
        private readonly GptEncoding _gptEncoding;
        private readonly int maxSequenceLength; // Max len para TRUNCAMENTO, não padding aqui

        /// <summary>
        /// Gets the token ID used for padding sequences.
        /// </summary>
        public int PadTokenId { get; private set; }

        /// <summary>
        /// Gets the token ID used for unknown tokens.
        /// </summary>
        public int UnkTokenId { get; private set; }

        /// <summary>
        /// Gets the token ID used for end of sequence.
        /// </summary>
        public int EosTokenId { get; private set; }

        /// <summary>
        /// Gets the size of the vocabulary used by the tokenizer.
        /// </summary>
        public int VocabSize { get; private set; }

        private const int StandardGpt2VocabSize = 50257;
        private const int StandardGpt2EosPadId = 50256;

        /// <summary>
        /// Initializes a new instance of the Tokenizer class.
        /// </summary>
        /// <param name="maxSequenceLength">The maximum sequence length for tokenization operations.</param>
        /// <exception cref="InvalidOperationException">Thrown when the GPT encoding initialization fails.</exception>
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

        /// <summary>
        /// Gets the actual vocabulary size used by the tokenizer.
        /// </summary>
        public int ActualVocabSize => this.VocabSize;

        /// <summary>
        /// Tokenizes the input text into a sequence of token IDs.
        /// </summary>
        /// <param name="text">The text to tokenize.</param>
        /// <param name="allowedSpecial">Optional set of special tokens that are allowed in the text.</param>
        /// <param name="applyPaddingTruncation">If true, applies padding or truncation to match maxSequenceLength.</param>
        /// <returns>An array of token IDs representing the tokenized text.</returns>
        /// <exception cref="InvalidOperationException">Thrown when the encoding is not initialized.</exception>
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


        /// <summary>
        /// Converts a sequence of token IDs back into text.
        /// </summary>
        /// <param name="tokens">The array of token IDs to convert.</param>
        /// <returns>The decoded text string.</returns>
        /// <exception cref="InvalidOperationException">Thrown when the encoding is not initialized.</exception>
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

        /// <summary>
        /// Gets the maximum sequence length configured for the tokenizer.
        /// </summary>
        /// <returns>The maximum sequence length.</returns>
        public int GetMaxSequenceLength()
        {
            return this.maxSequenceLength;
        }
    }
}