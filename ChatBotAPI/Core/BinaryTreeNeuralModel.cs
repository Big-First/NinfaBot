using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace ChatBotAPI.Core
{
    public class BinaryTreeNeuralModel
    {
        private Node _root;
        private readonly string _modelFilePath;
        private readonly Tokenizer _tokenizer;
        private readonly Random _random = new Random();

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

            // Remover tokens <|endoftext|> (50256) dos dados de treinamento
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

                Node nextNode = null;
                if (current.Token == token)
                {
                    nextNode = current;
                }
                else if (current.Left != null && current.Left.Token == token)
                {
                    nextNode = current.Left;
                }
                else if (current.Right != null && current.Right.Token == token)
                {
                    nextNode = current.Right;
                }
                else
                {
                    nextNode = new Node { Token = token };
                    if (current.Left == null)
                    {
                        current.Left = nextNode;
                    }
                    else if (current.Right == null)
                    {
                        current.Right = nextNode;
                    }
                    else
                    {
                        int leftCount = current.Left.NextTokens.Count;
                        int rightCount = current.Right.NextTokens.Count;
                        if (leftCount <= rightCount)
                        {
                            current.Left = nextNode;
                        }
                        else
                        {
                            current.Right = nextNode;
                        }
                    }
                }

                current = nextNode;
            }

            foreach (var targetToken in targetTokens)
            {
                if (!current.NextTokens.Contains(targetToken))
                    current.NextTokens.Add(targetToken);
            }

            Console.WriteLine(
                $"Training with input: [{string.Join(",", inputTokens)}], target: [{string.Join(",", targetTokens)}]");
            SaveModel();
        }

        public int GenerateNextToken(List<int> inputTokens, List<int> generatedTokens)
        {
            if (_root == null)
            {
                Console.WriteLine("Model is empty, using fallback.");
                return FallbackToken();
            }

            var current = _root;
            int depth = 0;
            const int maxDepth = 50;

            var cleanedInput = inputTokens.Where(t => t != 50256).ToList();
            if (cleanedInput.Count == 0)
            {
                Console.WriteLine("Input is empty after cleaning, using fallback.");
                return FallbackToken();
            }

            foreach (var token in cleanedInput)
            {
                Console.WriteLine(
                    $"Searching for token: {token}, current node: {(current != null ? current.Token : "null")}");
                if (depth++ > maxDepth || current == null)
                {
                    Console.WriteLine("Max depth reached or node not found during generation, using fallback.");
                    return FallbackToken();
                }

                if (current.Left != null && current.Left.Token == token)
                {
                    current = current.Left;
                }
                else if (current.Right != null && current.Right.Token == token)
                {
                    current = current.Right;
                }
                else
                {
                    current = null;
                }
            }

            if (current == null || current.NextTokens == null || current.NextTokens.Count == 0)
            {
                Console.WriteLine("Sequence not found, using fallback.");
                return FallbackToken();
            }

            int position = generatedTokens.Count % current.NextTokens.Count;
            int nextToken = current.NextTokens[position];
            Console.WriteLine($"Generated token: {nextToken} at position {generatedTokens.Count}");
            return nextToken;
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

        private void LoadModel()
        {
            if (File.Exists(_modelFilePath))
            {
                var json = File.ReadAllText(_modelFilePath);
                _root = JsonSerializer.Deserialize<Node>(json);
                Console.WriteLine("Model loaded from file.");
            }
        }

        private void SaveModel()
        {
            var json = JsonSerializer.Serialize(_root);
            File.WriteAllText(_modelFilePath, json);
            Console.WriteLine("Model saved to file.");
        }
    }
}