using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace ChatBotAPI.Core;

public class ChatBotService
{
    private readonly Tokenizer _tokenizer;
    private readonly BinaryTreeNeuralModel _model;
    private const int MaxSequenceLength = 128;
    private const int VocabSize = 50257;
    private const string ModelSavePath = "Vocabularys/model_tree.json";

    public ChatBotService()
    {
        var vocabPath = Path.Combine(Directory.GetCurrentDirectory(), "Vocabularys", "tokenizer.json");
        if (!File.Exists(vocabPath))
            throw new FileNotFoundException($"Tokenizer not found: {vocabPath}");

        _tokenizer = new Tokenizer(vocabPath);
        _model = new BinaryTreeNeuralModel(VocabSize, ModelSavePath);
        _model.LoadModel();

        var trainer = new Trainer(_tokenizer, _model);
        trainer.TrainAll();
        _model.SaveModel();

        Console.WriteLine("ChatBotService initialized!");
    }

    public string GetResponse(string message, int maxNewTokens = 100)
    {
        string normalizedMessage = message.Replace("What's", "What is").Replace("don't", "do not");
        List<int> inputTokens = _tokenizer.Encode(normalizedMessage);
        if (inputTokens.Count == 0 || inputTokens.All(id => id == 0))
            return "Could not process the message.";

        Console.WriteLine($"Input tokens: [{string.Join(",", inputTokens)}]");

        int[] inputTensorShape = new int[] { 1, MaxSequenceLength };
        long[] inputTensorData = new long[1 * MaxSequenceLength];
        for (int i = 0; i < inputTokens.Count && i < MaxSequenceLength; i++)
            inputTensorData[i] = inputTokens[i];
        var inputTensor = new SimpleTensor<long>(inputTensorData, inputTensorShape);

        List<int> generatedTokenIds = new List<int>();

        for (int step = 0; step < maxNewTokens; step++)
        {
            int nextTokenId = _model.GenerateNextToken(inputTensor, inputTokens.Count, generatedTokenIds);
            generatedTokenIds.Add(nextTokenId);

            if (nextTokenId == 13 || nextTokenId == 50256)
                break;
        }

        string response = _tokenizer.Decode(generatedTokenIds);
        Console.WriteLine($"Generated response: {response}");
        return response;
    }
}