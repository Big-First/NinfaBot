using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace ChatBotAPI.Core;

public class Tokenizer
{
    private readonly Dictionary<string, int> _tokenToId;
    private readonly Dictionary<int, string> _idToToken;
    private readonly List<(string, string)> _merges;
    private const int EndOfTextTokenId = 50256;

    public Tokenizer(string tokenizerFilePath)
    {
        _tokenToId = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        _idToToken = new Dictionary<int, string>();
        _merges = new List<(string, string)>();

        string json = File.ReadAllText(tokenizerFilePath);
        var tokenizerData = JsonSerializer.Deserialize<TokenizerData>(json)
                           ?? throw new InvalidOperationException("Failed to load tokenizer.json");

        foreach (var kvp in tokenizerData.model.vocab)
        {
            _tokenToId[kvp.Key] = kvp.Value;
            // Prefer non-Ġ version for decoding if ID is shared
            if (!_idToToken.ContainsKey(kvp.Value) || !kvp.Key.StartsWith("Ġ"))
            {
                _idToToken[kvp.Value] = kvp.Key;
            }
        }

        // Debug vocabulary
        var wordsToCheck = new[] { "The", "capital", "France", "Paris", "Hello", "Hi", "How", "can", "I", "help", "you", "today", "?", "I'm", "What's", "'m", "'s", "," };
        foreach (var word in wordsToCheck)
        {
            bool hasWord = _tokenToId.ContainsKey(word);
            bool hasGWord = _tokenToId.ContainsKey("Ġ" + word);
            Console.WriteLine($"Vocab contains '{word}': {hasWord} (ID: {(hasWord ? _tokenToId[word] : -1)})");
            Console.WriteLine($"Vocab contains 'Ġ{word}': {hasGWord} (ID: {(hasGWord ? _tokenToId["Ġ" + word] : -1)})");
        }

        var vocabOrdered = _tokenToId.OrderBy(kvp => kvp.Value).Select(kvp => kvp.Key).ToList();
        for (int i = 1; i < vocabOrdered.Count; i++)
        {
            var prev = vocabOrdered[i - 1];
            var curr = vocabOrdered[i];
            if (curr.StartsWith(prev, StringComparison.OrdinalIgnoreCase) && curr.Length > prev.Length)
            {
                _merges.Add((prev, curr.Substring(prev.Length)));
            }
        }

        Console.WriteLine($"Tokenizer initialized with {_tokenToId.Count} tokens.");
    }

    public List<int> Encode(string text)
    {
        if (string.IsNullOrEmpty(text))
            return new List<int>();

        var words = SplitWithPunctuation(text);
        Console.WriteLine($"Words extracted: [{string.Join(", ", words)}]");
        return EncodeWords(words);
    }

    private List<string> SplitWithPunctuation(string text)
    {
        var result = new List<string>();
        var currentWord = new StringBuilder();
        bool inContraction = false;

        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (c == '\'' && i + 1 < text.Length && char.IsLetter(text[i + 1]))
            {
                if (currentWord.Length > 0)
                {
                    result.Add(currentWord.ToString());
                    currentWord.Clear();
                }
                currentWord.Append(c);
                inContraction = true;
            }
            else if (inContraction && char.IsLetter(c))
            {
                currentWord.Append(c);
            }
            else if (char.IsPunctuation(c) || char.IsWhiteSpace(c))
            {
                if (currentWord.Length > 0)
                {
                    result.Add(currentWord.ToString());
                    currentWord.Clear();
                }
                if (!char.IsWhiteSpace(c))
                    result.Add(c.ToString());
                inContraction = false;
            }
            else
            {
                currentWord.Append(c);
                inContraction = false;
            }
        }
        if (currentWord.Length > 0)
            result.Add(currentWord.ToString());

        return result.Where(s => !string.IsNullOrWhiteSpace(s)).ToList();
    }

    private List<int> EncodeWords(List<string> words)
    {
        var tokenIds = new List<int>();

        foreach (var word in words)
        {
            if (_tokenToId.TryGetValue(word, out int id))
            {
                Console.WriteLine($"Mapped '{word}' → ID {id}");
                tokenIds.Add(id);
            }
            else if (word.EndsWith(".") || word.EndsWith(",") || word.EndsWith("?") || word.EndsWith("!"))
            {
                string baseWord = word.TrimEnd('.', ',', '?', '!');
                if (_tokenToId.TryGetValue(baseWord, out id))
                {
                    Console.WriteLine($"Mapped '{baseWord}' → ID {id}");
                    tokenIds.Add(id);
                    string punct = word.Substring(baseWord.Length);
                    if (_tokenToId.TryGetValue(punct, out int punctId))
                    {
                        Console.WriteLine($"Mapped '{punct}' → ID {punctId}");
                        tokenIds.Add(punctId);
                    }
                }
                else
                {
                    Console.WriteLine($"Word not found: {word}, trying fallback.");
                    tokenIds.AddRange(FallbackEncode(word));
                }
            }
            else
            {
                Console.WriteLine($"Word not found: {word}, trying fallback.");
                tokenIds.AddRange(FallbackEncode(word));
            }
        }

        Console.WriteLine($"Tokens generated: [{string.Join(",", tokenIds)}]");
        return tokenIds;
    }

    private List<int> FallbackEncode(string word)
    {
        var tokenIds = new List<int>();

        // Try contractions
        if (word.Contains("'"))
        {
            var parts = word.Split('\'', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 2)
            {
                string first = parts[0];
                string second = "'" + parts[1];
                if (_tokenToId.TryGetValue(first, out int firstId))
                {
                    Console.WriteLine($"Mapped '{first}' → ID {firstId}");
                    tokenIds.Add(firstId);
                }
                if (_tokenToId.TryGetValue(second, out int secondId))
                {
                    Console.WriteLine($"Mapped '{second}' → ID {secondId}");
                    tokenIds.Add(secondId);
                }
                if (tokenIds.Any())
                    return tokenIds;
            }
        }

        // Try lowercase
        if (_tokenToId.TryGetValue(word.ToLower(), out int id))
        {
            Console.WriteLine($"Mapped '{word.ToLower()}' → ID {id}");
            return new List<int> { id };
        }

        // Character-level fallback
        var tokens = word.ToCharArray().Select(c => c.ToString()).ToList();
        for (int i = 0; i < _merges.Count && tokens.Count > 1; i++)
        {
            var (first, second) = _merges[i];
            var newTokens = new List<string>();
            int j = 0;
            while (j < tokens.Count)
            {
                if (j + 1 < tokens.Count && tokens[j] == first && tokens[j + 1] == second)
                {
                    newTokens.Add(first + second);
                    j += 2;
                }
                else
                {
                    newTokens.Add(tokens[j]);
                    j++;
                }
            }
            tokens = newTokens;
        }

        foreach (var token in tokens)
        {
            if (_tokenToId.TryGetValue(token, out id))
            {
                Console.WriteLine($"Mapped token '{token}' → ID {id}");
                tokenIds.Add(id);
            }
            else
            {
                Console.WriteLine($"Token not found: {token}, using <unk>.");
                tokenIds.Add(0); // <unk>
            }
        }

        return tokenIds;
    }

    public string Decode(List<int> tokenIds)
    {
        if (tokenIds == null || tokenIds.Count == 0)
            return "";

        var words = new List<string>();
        foreach (var id in tokenIds)
        {
            if (_idToToken.TryGetValue(id, out string word))
            {
                Console.WriteLine($"Decoding ID {id} → '{word}'");
                words.Add(word);
            }
            else
            {
                Console.WriteLine($"ID {id} not found, using <unk>");
                words.Add("<unk>");
            }
        }

        var result = new StringBuilder();
        bool needsSpace = false;
        foreach (var word in words)
        {
            if (word.StartsWith("Ġ"))
            {
                if (needsSpace && !result.ToString().EndsWith(" "))
                    result.Append(" ");
                result.Append(word.Substring(1));
                needsSpace = true;
            }
            else if (word == "." || word == "," || word == "!" || word == "?")
            {
                result.Append(word);
                needsSpace = true;
            }
            else if (word == "'s" || word == "'m")
            {
                result.Append(word);
                needsSpace = false;
            }
            else
            {
                if (needsSpace && !result.ToString().EndsWith(" "))
                    result.Append(" ");
                result.Append(word);
                needsSpace = true;
            }
        }

        return result.ToString().Trim();
    }
}