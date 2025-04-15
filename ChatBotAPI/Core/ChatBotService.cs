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
    private static readonly string ModelSavePath = Path.Combine(Directory.GetCurrentDirectory(), "Vocabularys","model_tree.json"); // Confirmar nome

    public ChatBotService()
    {
        _tokenizer = new Tokenizer(VocabFilePath);
        _model = new BinaryTreeNeuralModel(_tokenizer, ModelSavePath);
        var trainer = new Trainer(_tokenizer, _model);
        trainer.TrainAll(); // Chamar o treinamento durante a inicialização
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

        var generatedTokens = GenerateResponse(inputTokens);
        var response = _tokenizer.Decode(generatedTokens);

        if (string.IsNullOrWhiteSpace(response))
        {
            return "I couldn't generate a response. Try asking something else!";
        }

        return response;
    }

    private List<int> GenerateResponse(List<int> inputTokens)
    {
        const int maxTokens = 20;
        var generatedTokens = new List<int>();
        bool hasPunctuation = false;

        for (int i = 0; i < maxTokens; i++)
        {
            var token = _model.GenerateNextToken(inputTokens, generatedTokens);
            if (token == -1)
            {
                Console.WriteLine("Generation failed, using fallback.");
                break;
            }

            generatedTokens.Add(token);
            Console.WriteLine($"Sampled token: {token} at position {i}");

            if (_tokenizer.Decode(new List<int> { token }) is string decodedToken &&
                (decodedToken == "." || decodedToken == "!" || decodedToken == "?"))
            {
                hasPunctuation = true;
                break;
            }
        }

        if (!hasPunctuation && generatedTokens.Count > 0)
        {
            generatedTokens.Add(13); // Adicionar um ponto final se não houver pontuação
        }

        return generatedTokens;
    }
}