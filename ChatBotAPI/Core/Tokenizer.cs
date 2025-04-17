using System;
using System.Collections.Generic;
using System.IO;
using System.Linq; // Necessário para OrderBy

namespace ChatBotAPI.Core
{
    public class Tokenizer
    {
        private readonly Dictionary<string, int> wordToIndex;
        private readonly Dictionary<int, string> indexToWord;
        public int PadTokenId { get; private set; }
        private readonly int maxSequenceLength;
        // vocabSizeLimit é o tamanho máximo permitido, não necessariamente o tamanho real
        public readonly int vocabSizeLimit;
        private readonly string padToken = "<PAD>";
        private readonly string unkToken = "<UNK>";
        public readonly int padTokenId;
        public readonly int unkTokenId;
        
        // Construtor Refatorado
        public Tokenizer(Dictionary<string, int>? loadedVocab, int maxSequenceLength, int vocabSizeLimit, string padToken = "<PAD>", string unkToken = "<UNK>")
        {
            // 1. Inicializa Dicionários PRIMEIRO
            wordToIndex = new Dictionary<string, int>();
            indexToWord = new Dictionary<int, string>();

            // 2. Atribui campos simples
            this.maxSequenceLength = maxSequenceLength;
            this.vocabSizeLimit = vocabSizeLimit;
            this.padToken = padToken ?? throw new ArgumentNullException(nameof(padToken));
            this.unkToken = unkToken ?? throw new ArgumentNullException(nameof(unkToken));

            // 3. Adiciona tokens especiais e DEFINE IDs (incluindo PadTokenId PÚBLICO)
            int index = 0;
            // Adiciona PAD
            Console.WriteLine($"DEBUG: Tokenizer Constructor - Adding PAD token: '{this.padToken}' with index {index}");
            wordToIndex.Add(this.padToken, index);
            indexToWord.Add(index, this.padToken);
            this.padTokenId = index; // Define campo privado
            this.PadTokenId = this.padTokenId; // *** Define propriedade PÚBLICA AQUI ***
            Console.WriteLine($"DEBUG: Tokenizer Constructor - Public PadTokenId set to {this.PadTokenId}"); // Confirma
            index++;

            // Adiciona UNK
            Console.WriteLine($"DEBUG: Tokenizer Constructor - Adding UNK token: '{this.unkToken}' with index {index}");
            wordToIndex.Add(this.unkToken, index);
            indexToWord.Add(index, this.unkToken);
            this.unkTokenId = index; // Define campo privado
             Console.WriteLine($"DEBUG: Tokenizer Constructor - unkTokenId set to {this.unkTokenId}"); // Confirma
            index++;

            // 4. Processa vocabulário carregado
            if (loadedVocab != null)
            {
                Console.WriteLine($"DEBUG: Tokenizer Constructor - Processing {loadedVocab.Count} entries from loaded vocab.");
                foreach (var word in loadedVocab.Keys)
                {
                    // Condição 1: Limite atingido? (vocabSizeLimit é grande, improvável)
                    if (index >= this.vocabSizeLimit) {
                        Console.WriteLine($"DEBUG: Vocab limit reached ({this.vocabSizeLimit})"); // Adicione log se suspeitar
                        break;
                    }
                    // Condição 2: Palavra é vazia? (Improvável para todas as 740k)
                    if (string.IsNullOrWhiteSpace(word)) {
                        // Console.WriteLine("DEBUG: Skipping whitespace word"); // Log opcional
                        continue;
                    }
                    // Condição 3: Palavra é PAD ou UNK? (Só deveria pular duas)
                    if (word == this.padToken || word == this.unkToken) {
                        // Console.WriteLine("DEBUG: Skipping special token"); // Log opcional
                        continue;
                    }

                    // Condição 4: Palavra já existe? (Só aconteceria se o JSON tivesse duplicatas EXATAS)
                    if (!wordToIndex.ContainsKey(word))
                    {
                        wordToIndex.Add(word, index);
                        indexToWord.Add(index, word);
                        // *** ESTE INCREMENTO ESTÁ SENDO CHAMADO? ***
                        index++; // <-- Se esta linha nunca for atingida, ActualVocabSize ficará em 2.
                    } else {
                        // Console.WriteLine($"DEBUG: Word duplicate: {word}"); // Log opcional
                    }
                } // Fim do foreach
            }
            else { Console.Error.WriteLine("Warning: Loaded vocabulary dictionary was null."); }

            // 5. Define ActualVocabSize FINAL
            this.ActualVocabSize = index; // Atualiza com o índice final
            Console.WriteLine($"Tokenizer initialized. Actual Vocab Size: {this.ActualVocabSize}, PadTokenId = {this.PadTokenId}"); // Log final
        }

        public int GetMaxSequenceLength()
        {
            // Retorna o valor do campo privado que foi definido no construtor
            return this.maxSequenceLength;
        }
        // Propriedade para saber o tamanho real do vocabulário usado
        public int ActualVocabSize { get; private set; }


        // Remover método LoadConfig, pois a lógica está agora no construtor
        // private void LoadConfig(string configPath) { ... }

        public int[] Tokenize(string text)
        {
            string[] words = text.ToLower().Split(' ', StringSplitOptions.RemoveEmptyEntries);
            List<int> tokens = new List<int>();

            foreach (string word in words)
            {
                if (wordToIndex.TryGetValue(word, out int index))
                {
                    tokens.Add(index);
                }
                else
                {
                    tokens.Add(this.unkTokenId); // Use o ID UNK armazenado
                }
            }

            // Padding
            while (tokens.Count < maxSequenceLength)
            {
                tokens.Add(this.padTokenId); // Use o ID PAD armazenado
            }
            // Truncating
            if (tokens.Count > maxSequenceLength)
            {
                tokens = tokens.GetRange(0, maxSequenceLength);
            }

            return tokens.ToArray();
        }

        public string Detokenize(int[] tokens)
        {
            List<string> words = new List<string>();
            foreach (int token in tokens)
            {
                // Usa os IDs armazenados para ignorar tokens especiais
                if (token != this.padTokenId && indexToWord.TryGetValue(token, out string word))
                {
                     // Opcional: também pode querer ignorar UNK na detokenização
                     // if (token != this.padTokenId && token != this.unkTokenId && indexToWord.TryGetValue(token, out string word))
                    words.Add(word);
                }
                // Se quiser mostrar UNK explicitamente:
                // else if (token == this.unkTokenId)
                // {
                //     words.Add(this.unkToken);
                // }
            }
            return string.Join(" ", words);
        }
    }
}