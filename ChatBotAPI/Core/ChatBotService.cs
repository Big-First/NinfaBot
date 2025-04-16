using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace ChatBotAPI.Core;

public class ChatBotService
{
    private readonly BinaryTreeNeuralModel _model;
    private readonly Tokenizer _tokenizer;
    private static readonly string VocabFilePath = Path.Combine(Directory.GetCurrentDirectory(), "Vocabularys", "tokenizer.json");
    private static readonly string ModelFilePath = Path.Combine(Directory.GetCurrentDirectory(), "Vocabularys","model_tree.json"); // Confirmar nome

    public ChatBotService()
    {
        _tokenizer = new Tokenizer(VocabFilePath);
        _model = new BinaryTreeNeuralModel(_tokenizer, ModelFilePath);
        var trainer = new Trainer(_tokenizer, _model);
        trainer.TrainAll();
        Console.WriteLine("ChatBotService initialized!");
    }

    public string GetResponse(string message)
    {
        if (string.IsNullOrWhiteSpace(message))
        {
            return "Please enter a message.";
        }

        var inputTokens = _tokenizer.Encode(message);
        Console.WriteLine($"Input tokens: [{string.Join(",", inputTokens)}]");

        var generatedTokens = _model.GenerateResponse(inputTokens);
        var response = _tokenizer.Decode(generatedTokens);

        if (string.IsNullOrWhiteSpace(response))
        {
            return "I couldn't generate a response. Try asking something else!";
        }

        Console.WriteLine($"Response: {response}");
        return response;
    }
}