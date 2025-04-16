using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace ChatBotAPI.Core
{
    public class Tokenizer
    {
        private Dictionary<string, int> wordToIndex;
        private Dictionary<int, string> indexToWord;
        private readonly int maxSequenceLength;
        private readonly int vocabSize;
        private readonly string padToken = "<PAD>";
        private readonly string unkToken = "<UNK>";

        public Tokenizer(string configPath, int maxSequenceLength, int vocabSize)
        {
            this.maxSequenceLength = maxSequenceLength;
            this.vocabSize = vocabSize;
            wordToIndex = new Dictionary<string, int>();
            indexToWord = new Dictionary<int, string>();

            LoadConfig(configPath);
        }

        private void LoadConfig(string configPath)
        {
            if (!File.Exists(configPath))
            {
                throw new FileNotFoundException($"Tokenizer config file not found at: {configPath}");
            }

            string json = File.ReadAllText(configPath);
            try
            {
                var modelConfig = JsonSerializer.Deserialize<Model>(json);
                if (modelConfig?.Vocab == null)
                {
                    throw new JsonException($"Tokenizer config is missing a valid 'Vocab' field at: {configPath}");
                }

                int index = 0;
                wordToIndex.Add(padToken, index);
                indexToWord.Add(index, padToken);
                index++;

                wordToIndex.Add(unkToken, index);
                indexToWord.Add(index, unkToken);
                index++;

                foreach (var pair in modelConfig.Vocab)
                {
                    if (index >= vocabSize) break;
                    if (string.IsNullOrWhiteSpace(pair.Key)) continue;
                    wordToIndex[pair.Key] = index;
                    indexToWord[index] = pair.Key;
                    index++;
                }
            }
            catch (JsonException ex)
            {
                throw new JsonException($"Failed to deserialize tokenizer config. Ensure the JSON contains 'Type' and 'Vocab' fields. Path: {configPath}, Content: {json}", ex);
            }
        }

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
                    tokens.Add(wordToIndex[unkToken]);
                }
            }

            while (tokens.Count < maxSequenceLength)
            {
                tokens.Add(wordToIndex[padToken]);
            }
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
                if (indexToWord.TryGetValue(token, out string word) && word != padToken)
                {
                    words.Add(word);
                }
            }
            return string.Join(" ", words);
        }
    }
}