using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace ChatBotAPI.Core
{
    public class BinaryTreeNeuralModel
    {
        public Node _root;
        public readonly string _modelFilePath;
        public readonly Tokenizer _tokenizer;
        public readonly Random _random = new Random();

        public BinaryTreeNeuralModel(Tokenizer tokenizer, string modelFilePath)
        {
            _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
            _modelFilePath = modelFilePath ?? throw new ArgumentNullException(nameof(modelFilePath));
            LoadModel();
        }

        public void Train(List<int> inputTokens, List<int> targetTokens)
        {
            if (inputTokens == null || inputTokens.Count == 0 || targetTokens == null || targetTokens.Count == 0)
            {
                Console.WriteLine("Invalid training data, skipping.");
                return;
            }

            inputTokens = inputTokens.Where(t => t != 50256).ToList();
            targetTokens = targetTokens.Where(t => t != 50256).ToList();

            if (inputTokens.Count == 0 || targetTokens.Count == 0)
            {
                Console.WriteLine("Training data is empty after cleaning, skipping.");
                return;
            }

            if (_root == null)
            {
                _root = new Node { Token = inputTokens[0] };
                Console.WriteLine($"Root node created with token: {inputTokens[0]}");
            }

            var current = _root;
            int depth = 0;
            const int maxDepth = 50;

            foreach (var token in inputTokens)
            {
                if (depth++ > maxDepth)
                {
                    Console.WriteLine("Max depth reached during training, stopping.");
                    break;
                }

                var nextNode = current.Children.FirstOrDefault(c => c.Token == token);
                if (nextNode == null)
                {
                    nextNode = new Node { Token = token };
                    current.Children.Add(nextNode);
                }

                current = nextNode;
            }

            // Adicionar a sequência de resposta completa
            if (!current.ResponseSequences.Any(seq => seq.SequenceEqual(targetTokens)))
            {
                current.ResponseSequences.Add(targetTokens);
            }

            Console.WriteLine(
                $"Training with input: [{string.Join(",", inputTokens)}], target: [{string.Join(",", targetTokens)}]");
            LogTree(_root); // Adicionar depuração da árvore
            SaveModel();
        }

        public List<int> GenerateResponse(List<int> inputTokens)
        {
            if (_root == null)
            {
                Console.WriteLine("Model is empty, using fallback.");
                return new List<int> { FallbackToken() };
            }

            var cleanedInput = inputTokens.Where(t => t != 50256).ToList();
            if (cleanedInput.Count == 0)
            {
                Console.WriteLine("Input is empty after cleaning, using fallback.");
                return new List<int> { FallbackToken() };
            }

            var current = _root;
            int depth = 0;
            const int maxDepth = 50;

            // Buscar o caminho na árvore
            foreach (var token in cleanedInput)
            {
                Console.WriteLine(
                    $"Searching for token: {token}, current node: {(current != null ? current.Token : "null")}");
                if (depth++ > maxDepth || current == null)
                {
                    Console.WriteLine("Max depth reached or node not found during generation, using fallback.");
                    return new List<int> { FallbackToken() };
                }

                current = current.Children.FirstOrDefault(c => c.Token == token);
            }

            if (current == null || current.ResponseSequences == null || current.ResponseSequences.Count == 0)
            {
                Console.WriteLine("Sequence not found, using fallback.");
                return new List<int> { FallbackToken() };
            }

            // Escolher aleatoriamente uma sequência de resposta completa
            var selectedSequence = current.ResponseSequences[_random.Next(current.ResponseSequences.Count)];
            Console.WriteLine($"Selected response sequence: [{string.Join(",", selectedSequence)}]");
            return selectedSequence;
        }

        private int FallbackToken()
        {
            var allTokens = _tokenizer.GetAllTokenIds();
            if (allTokens.Count == 0)
            {
                Console.WriteLine("No tokens available in vocabulary, returning -1.");
                return -1;
            }

            int token = allTokens[_random.Next(allTokens.Count)];
            Console.WriteLine($"Fallback token selected: {token}");
            return token;
        }

        public void LogTree(Node node, int depth = 0)
        {
            if (node == null) return;
            Console.WriteLine(new string(' ', depth * 2) + $"Token: {node.Token}");
            foreach (var sequence in node.ResponseSequences)
            {
                Console.WriteLine(new string(' ', depth * 2 + 2) + $"Response: [{string.Join(",", sequence)}]");
            }

            foreach (var child in node.Children)
            {
                LogTree(child, depth + 1);
            }
        }

        public void LoadModel()
        {
            if (File.Exists(_modelFilePath))
            {
                var json = File.ReadAllText(_modelFilePath);
                _root = JsonSerializer.Deserialize<Node>(json);
                Console.WriteLine("Model loaded from file.");
            }
        }

        public void SaveModel()
        {
            var json = JsonSerializer.Serialize(_root);
            File.WriteAllText(_modelFilePath, json);
            Console.WriteLine("Model saved to file.");
        }
    }
}