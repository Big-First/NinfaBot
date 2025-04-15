using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;

namespace ChatBotAPI.Core;

public class BinaryTreeNeuralModel
{

    private TreeNode? _root;
    private readonly int _vocabSize;
    private readonly string _savePath;
    private readonly Random _random = new Random();
    private const float Temperature = 0.7f;

    public BinaryTreeNeuralModel(int vocabSize, string savePath)
    {
        _vocabSize = vocabSize;
        _savePath = savePath;
    }

    public void Train(List<int> inputTokens, List<int> targetTokens)
    {
        if (inputTokens == null || targetTokens == null || inputTokens.Count == 0 || targetTokens.Count == 0)
        {
            Console.WriteLine("Invalid input or target tokens, skipping training.");
            return;
        }

        Console.WriteLine($"Training with input: [{string.Join(",", inputTokens)}], target: [{string.Join(",", targetTokens)}]");
        bool isCapitalQuery = inputTokens.Contains(6818);
        _root = InsertOrUpdate(_root, inputTokens, targetTokens, isCapitalQuery);
        SaveModel();
    }

    private TreeNode InsertOrUpdate(TreeNode? node, List<int> inputTokens, List<int> targetTokens, bool isDeterministic)
    {
        if (node == null)
        {
            node = new TreeNode
            {
                InputSequence = inputTokens.ToList(),
                TargetSequences = new List<(List<int>, float)> { (targetTokens.ToList(), 1.0f) },
                IsDeterministic = isDeterministic
            };
            UpdateProbabilities(node);
            return node;
        }

        string inputStr = string.Join(",", inputTokens);
        string nodeStr = string.Join(",", node.InputSequence);
        int comparison = string.Compare(inputStr, nodeStr, StringComparison.Ordinal);

        if (comparison < 0)
            node.Left = InsertOrUpdate(node.Left, inputTokens, targetTokens, isDeterministic);
        else if (comparison > 0)
            node.Right = InsertOrUpdate(node.Right, inputTokens, targetTokens, isDeterministic);
        else
        {
            var existing = node.TargetSequences.FirstOrDefault(t => t.Sequence.SequenceEqual(targetTokens));
            if (existing.Sequence != null)
            {
                node.TargetSequences[node.TargetSequences.IndexOf(existing)] = (existing.Sequence, existing.Weight + 1.0f);
            }
            else
            {
                node.TargetSequences.Add((targetTokens.ToList(), 1.0f));
            }
            node.IsDeterministic = node.IsDeterministic || isDeterministic;
            UpdateProbabilities(node);
        }

        return node;
    }

    private void UpdateProbabilities(TreeNode node)
    {
        node.NextTokenProbs.Clear();
        if (node.IsDeterministic)
        {
            var sequence = node.TargetSequences.First().Sequence;
            for (int i = 0; i < sequence.Count; i++)
            {
                var probs = new Dictionary<int, float>();
                probs[sequence[i]] = 1.0f;
                node.NextTokenProbs.Add(probs);
            }
            return;
        }

        int maxLength = node.TargetSequences.Max(s => s.Sequence.Count);
        for (int i = 0; i < maxLength; i++)
        {
            var tokenCounts = new Dictionary<int, float>();
            float totalWeight = 0f;
            foreach (var (sequence, weight) in node.TargetSequences)
            {
                if (i < sequence.Count)
                {
                    int token = sequence[i];
                    tokenCounts[token] = tokenCounts.GetValueOrDefault(token, 0f) + weight;
                    totalWeight += weight;
                }
            }

            var probs = new Dictionary<int, float>();
            foreach (var kvp in tokenCounts)
            {
                probs[kvp.Key] = kvp.Value / totalWeight;
            }
            node.NextTokenProbs.Add(probs);
        }
    }

    public int GenerateNextToken(SimpleTensor<long> inputTensor, int currentLength, List<int> generatedTokens)
    {
        if (inputTensor == null || currentLength <= 0 || inputTensor.Data.All(x => x == 0))
        {
            Console.WriteLine("Invalid input tensor or length <= 0, returning period.");
            return 13;
        }

        List<int> inputTokens = new List<int>();
        for (int i = 0; i < currentLength && i < inputTensor.Data.Length; i++)
        {
            if (inputTensor.Data[i] != 0)
                inputTokens.Add((int)inputTensor.Data[i]);
        }

        if (!inputTokens.Any())
        {
            Console.WriteLine("No valid input tokens, returning period.");
            return 13;
        }

        TreeNode? node = FindNode(_root, inputTokens);
        if (node == null || node.NextTokenProbs.Count == 0)
        {
            Console.WriteLine($"No node found for input: [{string.Join(",", inputTokens)}], returning period.");
            return 13;
        }

        int position = generatedTokens.Count;
        if (position >= node.NextTokenProbs.Count)
        {
            Console.WriteLine($"End of sequence for input: [{string.Join(",", inputTokens)}], returning period.");
            return 13;
        }

        var probs = node.NextTokenProbs[position];
        if (probs.Count == 0)
        {
            Console.WriteLine($"No tokens available at position {position}, returning period.");
            return 13;
        }

        int nextToken = SampleToken(probs);
        Console.WriteLine($"Sampled token: {nextToken} at position {position}");
        return nextToken;
    }

    private int SampleToken(Dictionary<int, float> probs)
    {
        var scaledProbs = new Dictionary<int, float>();
        float sum = 0f;
        foreach (var kvp in probs)
        {
            float scaled = (float)Math.Pow(kvp.Value, 1.0 / Temperature);
            scaledProbs[kvp.Key] = scaled;
            sum += scaled;
        }

        foreach (var key in scaledProbs.Keys.ToList())
        {
            scaledProbs[key] /= sum;
        }

        float r = (float)_random.NextDouble();
        float cumulative = 0f;
        foreach (var kvp in scaledProbs)
        {
            cumulative += kvp.Value;
            if (r <= cumulative)
            {
                return kvp.Key;
            }
        }

        return scaledProbs.OrderByDescending(kvp => kvp.Value).First().Key;
    }

    private TreeNode? FindNode(TreeNode? node, List<int> inputTokens)
    {
        if (node == null) return null;

        string inputStr = string.Join(",", inputTokens);
        string nodeStr = string.Join(",", node.InputSequence);

        if (inputStr == nodeStr)
            return node;

        int comparison = string.Compare(inputStr, nodeStr, StringComparison.Ordinal);
        if (comparison < 0)
            return FindNode(node.Left, inputTokens);
        return FindNode(node.Right, inputTokens);
    }

    public void SaveModel()
    {
        try
        {
            var serializedTree = SerializeTree(_root);
            File.WriteAllText(_savePath, JsonSerializer.Serialize(serializedTree, new JsonSerializerOptions { WriteIndented = true }));
            Console.WriteLine($"Model saved to {_savePath}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to save model: {ex.Message}");
        }
    }

    public void LoadModel()
    {
        if (!File.Exists(_savePath))
        {
            Console.WriteLine($"No model file found at {_savePath}, starting with empty model.");
            return;
        }

        try
        {
            var json = File.ReadAllText(_savePath);
            if (string.IsNullOrWhiteSpace(json))
            {
                Console.WriteLine($"Model file at {_savePath} is empty, starting with empty model.");
                return;
            }

            var serializedTree = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(json);
            _root = DeserializeTree(serializedTree);
            Console.WriteLine($"Model loaded from {_savePath}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to load model from {_savePath}: {ex.Message}");
            Console.WriteLine("Starting with empty model to prevent crash.");
            _root = null;
        }
    }

    private Dictionary<string, object> SerializeTree(TreeNode? node)
    {
        var dict = new Dictionary<string, object>();
        if (node == null) return dict;

        dict["InputSequence"] = node.InputSequence;
        dict["TargetSequences"] = node.TargetSequences.Select(t => new { Sequence = t.Sequence, Weight = t.Weight }).ToList();
        dict["NextTokenProbs"] = node.NextTokenProbs;
        dict["IsDeterministic"] = node.IsDeterministic;
        dict["Left"] = SerializeTree(node.Left);
        dict["Right"] = SerializeTree(node.Right);
        return dict;
    }

    private TreeNode? DeserializeTree(Dictionary<string, JsonElement>? dict)
    {
        if (dict == null || !dict.ContainsKey("InputSequence"))
        {
            Console.WriteLine("Invalid or empty dictionary in DeserializeTree, returning null.");
            return null;
        }

        var node = new TreeNode
        {
            InputSequence = dict.TryGetValue("InputSequence", out var inputSeq)
                ? JsonSerializer.Deserialize<List<int>>(inputSeq.GetRawText()) ?? new List<int>()
                : new List<int>(),
            IsDeterministic = dict.TryGetValue("IsDeterministic", out var isDet)
                ? isDet.GetBoolean()
                : false,
            TargetSequences = new List<(List<int>, float)>(),
            NextTokenProbs = new List<Dictionary<int, float>>()
        };

        // Handle TargetSequences (new format) or TargetSequence (old format)
        if (dict.TryGetValue("TargetSequences", out var targetSeqs))
        {
            try
            {
                node.TargetSequences = JsonSerializer.Deserialize<List<(List<int>, float)>>(targetSeqs.GetRawText())
                                      ?? new List<(List<int>, float)>();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to deserialize TargetSequences: {ex.Message}");
            }
        }
        else if (dict.TryGetValue("TargetSequence", out var oldTargetSeq))
        {
            // Backward compatibility: Convert old TargetSequence to TargetSequences
            try
            {
                var sequence = JsonSerializer.Deserialize<List<int>>(oldTargetSeq.GetRawText());
                if (sequence != null)
                {
                    node.TargetSequences.Add((sequence, 1.0f));
                    Console.WriteLine("Converted old TargetSequence to TargetSequences for backward compatibility.");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to convert old TargetSequence: {ex.Message}");
            }
        }

        // Handle NextTokenProbs (optional in old format)
        if (dict.TryGetValue("NextTokenProbs", out var probs))
        {
            try
            {
                node.NextTokenProbs = JsonSerializer.Deserialize<List<Dictionary<int, float>>>(probs.GetRawText())
                                     ?? new List<Dictionary<int, float>>();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to deserialize NextTokenProbs: {ex.Message}");
            }
        }

        // Update probabilities if TargetSequences exist
        if (node.TargetSequences.Any())
        {
            UpdateProbabilities(node);
        }

        node.Left = dict.TryGetValue("Left", out var leftElement) && leftElement.ValueKind != JsonValueKind.Null
            ? DeserializeTree(JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(leftElement.GetRawText()))
            : null;
        node.Right = dict.TryGetValue("Right", out var rightElement) && rightElement.ValueKind != JsonValueKind.Null
            ? DeserializeTree(JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(rightElement.GetRawText()))
            : null;

        return node;
    }
}