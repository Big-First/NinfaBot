using System;
using System.Collections.Generic;
using System.Linq;

namespace ChatBotAPI.Core;

public class Trainer
{
    private readonly Tokenizer _tokenizer;
    private readonly BinaryTreeNeuralModel _model;

    public Trainer(Tokenizer tokenizer, BinaryTreeNeuralModel model)
    {
        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
        _model = model ?? throw new ArgumentNullException(nameof(model));
    }

    public void TrainAll()
    {
        var trainingData = GetTrainingData();
        int successfulPairs = 0;

        foreach (var (input, output) in trainingData)
        {
            string normalizedInput = input.Replace("What's", "What is").Replace("don't", "do not");
            List<int> inputTokens = _tokenizer.Encode(normalizedInput);
            List<int> outputTokens = _tokenizer.Encode(output);

            if (inputTokens.Count == 0 || outputTokens.Count == 0 || inputTokens.Contains(0) || outputTokens.Contains(0))
            {
                Console.WriteLine($"Warning: Invalid tokens for input '{normalizedInput}' or output '{output}'. Input: [{string.Join(",", inputTokens)}], Output: [{string.Join(",", outputTokens)}]. Skipping...");
                continue;
            }

            Console.WriteLine($"Training: '{normalizedInput}' → '{output}'");
            Console.WriteLine($"Input tokens: [{string.Join(",", inputTokens)}]");
            Console.WriteLine($"Output tokens: [{string.Join(",", outputTokens)}]");

            _model.Train(inputTokens, outputTokens);
            successfulPairs++;
        }

        Console.WriteLine($"Training completed: {successfulPairs}/{trainingData.Count} pairs processed.");
    }

    private static List<(string input, string output)> GetTrainingData()
    {
        return new List<(string input, string output)>
        {
            ("The capital of France is", "Paris."),
            ("The capital of Brazil is", "Brasília."),
            ("The capital of Japan is", "Tokyo."),
            ("The capital of Australia is", "Canberra."),
            ("The capital of Canada is", "Ottawa."),
            ("The capital of Germany is", "Berlin."),
            ("The capital of Italy is", "Rome."),
            ("Hello", "Hi! How can I help you today?"),
            ("Hello", "Hello! What's on your mind?"),
            ("Hello", "Hey there! Nice to see you!"),
            ("Hi", "Hello! What's on your mind?"),
            ("Hi", "Hi! How can I assist you?"),
            ("Good morning", "Good morning to you too!"),
            ("Hey there", "Hey! Nice to hear from you."),
            ("What is the largest planet?", "Jupiter is the largest planet in our solar system."),
            ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
            ("What is the boiling point of water?", "The boiling point of water is 100 degrees Celsius at standard pressure."),
            ("What is the tallest mountain?", "Mount Everest is the tallest mountain in the world."),
            ("Who are you?", "I'm a chatbot created to answer your questions!"),
            ("What can you do?", "I can answer questions about capitals, general knowledge, and chat about various topics!"),
            ("What's your name?", "I'm Ninfa, nice to meet you!"),
            ("Tell me something interesting", "Did you know that octopuses have three hearts?"),
            ("I don't know what to ask", "No worries! How about asking me about a country's capital or a fun fact?"),
            ("What's up?", "Not much, just hanging out in the digital world. What's up with you?"),
            ("Help", "I'm here to help! Try asking about a capital city or a general knowledge question."),
            ("Tell me about France", "France is known for its culture, wine, and the Eiffel Tower in Paris."),
            ("More about France", "France has a rich history with landmarks like the Louvre and Versailles."),
            ("Tell me about Brazil", "Brazil is famous for Carnival, the Amazon rainforest, and soccer."),
            ("More about Brazil", "Brazil's capital is Brasília, and it has vibrant cities like Rio de Janeiro.")
        };
    }
}