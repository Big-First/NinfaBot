// Tokenizer.cs - Final Version using SharpToken for GPT-2

using System;
using System.Collections.Generic;
using System.Linq;
using SharpToken; // Ensure this using is present

namespace ChatBotAPI.Core
{
    public partial class Tokenizer
    {
        private readonly GptEncoding _gptEncoding;

        // --- IDs ---
        // Define based on known GPT-2 / r50k_base standard values
        public int PadTokenId { get; private set; }
        public int UnkTokenId { get; private set; } // Not applicable for GPT-2 BPE standard usage
        public int EosTokenId { get; private set; }
        public int VocabSize { get; private set; }

        // --- Standard GPT-2 Values ---
        private const int StandardGpt2VocabSize = 50257;
        private const int StandardGpt2EosPadId = 50256; // <|endoftext|>

        private readonly int maxSequenceLength;

        // --- Constructor using SharpToken ---
        public Tokenizer(int maxSequenceLength)
        {
            this.maxSequenceLength = maxSequenceLength;
            Console.WriteLine($"Initializing Tokenizer using SharpToken for GPT-2 (r50k_base) encoding.");

            try
            {
                // 1. Get the correct encoding for GPT-2
                //    Use "r50k_base" as identified for GPT-2 standard encoding
                const string encodingName = "r50k_base";
                _gptEncoding = GptEncoding.GetEncoding(encodingName);

                if (_gptEncoding == null)
                {
                    throw new InvalidOperationException($"Could not get '{encodingName}' encoding from SharpToken library.");
                }
                Console.WriteLine($"Successfully obtained '{_gptEncoding}' encoding from SharpToken.");

                // 2. Set IDs and VocabSize based on KNOWN GPT-2 STANDARD VALUES
                //    Your TorchSharp model needs these exact values.
                this.PadTokenId = StandardGpt2EosPadId; // 50256
                this.EosTokenId = StandardGpt2EosPadId; // 50256
                this.UnkTokenId = -1;                   // Indicate N/A
                this.VocabSize = StandardGpt2VocabSize; // 50257

                Console.WriteLine($"Tokenizer initialized using SharpToken ('{_gptEncoding}').");
                Console.WriteLine($"--> Using Standard GPT-2 Values for Model Config:");
                Console.WriteLine($"    Vocab Size: {this.VocabSize}");
                Console.WriteLine($"    Pad Token ID: {this.PadTokenId}");
                Console.WriteLine($"    EOS Token ID: {this.EosTokenId}");
                Console.WriteLine($"    UNK Token ID: {this.UnkTokenId} (N/A)");
            }
            catch (ArgumentException ex) // Catch specific error for unknown encoding
            {
                 Console.Error.WriteLine($"FATAL ERROR: SharpToken does not recognize the encoding name used: {ex.Message}");
                 Console.Error.WriteLine("Ensure the 'SharpToken' package is up-to-date and the encoding name is correct (e.g., 'r50k_base').");
                 throw;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"FATAL ERROR during Tokenizer initialization with SharpToken: {ex.ToString()}");
                throw;
            }
        }

        // Property ActualVocabSize (returns the standard size)
        public int ActualVocabSize => this.VocabSize;

        // Method Tokenize
        public int[] Tokenize(string text)
        {
            if (_gptEncoding == null) throw new InvalidOperationException("SharpToken encoding not initialized.");
            try
            {
                List<int> tokens = _gptEncoding.Encode(text);

                // Apply Truncation and Padding MANUALLY
                int currentLength = tokens.Count;
                if (currentLength > maxSequenceLength)
                {
                    tokens = tokens.GetRange(0, maxSequenceLength);
                    // Console.WriteLine($"Warning: Input text truncated to {maxSequenceLength} tokens.");
                }
                else if (currentLength < maxSequenceLength)
                {
                    // Pad with the EOS/PAD ID
                    tokens.AddRange(Enumerable.Repeat(this.PadTokenId, maxSequenceLength - currentLength));
                }
                return tokens.ToArray();
            }
            catch (Exception ex) { Console.Error.WriteLine($"SharpToken Tokenize Error: {ex}"); throw; }
        }

        // Method Detokenize
        public string Detokenize(int[] tokens)
        {
             if (_gptEncoding == null) throw new InvalidOperationException("SharpToken encoding not initialized.");
             try
             {
                 // Filter PAD/EOS IDs before decoding
                 List<int> idsToDecode = tokens.Where(t => t != this.PadTokenId).ToList();
                 if (idsToDecode.Count == 0) return "";

                 // Decode using SharpToken
                 string decodedText = _gptEncoding.Decode(idsToDecode);
                 return decodedText.Trim();
             }
             catch (Exception ex) { Console.Error.WriteLine($"SharpToken Detokenize Error: {ex}"); return "[Detokenization Error]"; }
        }

        // Method GetMaxSequenceLength
        public int GetMaxSequenceLength()
        {
            return this.maxSequenceLength;
        }
    }
}