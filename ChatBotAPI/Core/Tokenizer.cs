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
        private readonly int padTokenId;
        private readonly int unkTokenId;

        // Construtor Refatorado
        public Tokenizer(Dictionary<string, int> loadedVocab, int maxSequenceLength, int vocabSizeLimit, string padToken = "<PAD>", string unkToken = "<UNK>")
        {
            this.maxSequenceLength = maxSequenceLength;
            this.vocabSizeLimit = vocabSizeLimit; // Limite máximo
            this.padToken = padToken;
            this.unkToken = unkToken;

            wordToIndex = new Dictionary<string, int>();
            indexToWord = new Dictionary<int, string>();

            // Inicializa com tokens especiais garantidos
            int index = 0;
            wordToIndex.Add(this.padToken, index);
            indexToWord.Add(index, this.padToken);
            this.padTokenId = index;
            index++;

            wordToIndex.Add(this.unkToken, index);
            indexToWord.Add(index, this.unkToken);
            this.unkTokenId = index;
            index++;

            if (loadedVocab == null)
            {
                // Decide how to handle this - maybe throw an exception or log a warning
                // For now, we'll just proceed with only PAD and UNK
                 Console.Error.WriteLine("Warning: Loaded vocabulary was null. Tokenizer initialized with only PAD and UNK tokens.");
            }
            else
            {
                // Adiciona vocabulário carregado, respeitando o limite e excluindo tokens especiais se já existirem
                // Ordenar pode ajudar na consistência se o dicionário original não tiver ordem garantida,
                // mas o Vocab do JSON geralmente não tem uma ordem intrínseca.
                // Usar os índices do JSON original é geralmente preferível se eles existirem e forem significativos.
                // Se o loadedVocab já tem os índices corretos (como no JSON), podemos usá-los,
                // mas precisamos garantir que não colidam e estejam dentro do limite.

                // Abordagem mais segura: Reconstruir índices a partir do vocabulário carregado
                // (ignorando os índices originais do JSON para garantir consistência interna aqui)
                foreach (var word in loadedVocab.Keys)
                {
                    if (index >= this.vocabSizeLimit) break; // Respeita o limite
                    if (string.IsNullOrWhiteSpace(word)) continue;
                    if (word == this.padToken || word == this.unkToken) continue; // Evita duplicatas

                    if (!wordToIndex.ContainsKey(word)) // Garante que não adicionamos duplicatas
                    {
                         wordToIndex[word] = index;
                         indexToWord[index] = word;
                         index++;
                    }
                }
            }

             // O tamanho real do vocabulário usado pelo tokenizer
             this.ActualVocabSize = index;
             Console.WriteLine($"Tokenizer initialized. Actual Vocab Size: {this.ActualVocabSize}, Limit: {this.vocabSizeLimit}");

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