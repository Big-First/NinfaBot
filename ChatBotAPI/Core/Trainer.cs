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
        Console.WriteLine("Iniciando treinamento...");
        var trainingData = GetTrainingData();
        int successfulPairs = 0;
        int pairCount = 0;

        foreach (var (input, output) in trainingData)
        {
            pairCount++;
            Console.WriteLine($"\n--- Processando Par de Treinamento {pairCount}/{trainingData.Count} ---");
            // Normalização simples (pode ser expandida)
            string normalizedInput = input.Replace("What's", "What is")
                                          .Replace("don't", "do not")
                                          .Replace("I'm", "I am"); // Adiciona mais se necessário

            Console.WriteLine($"Entrada Normalizada: '{normalizedInput}'");
            Console.WriteLine($"Saída Original: '{output}'");

            List<int> inputTokens = _tokenizer.Encode(normalizedInput);
            List<int> outputTokens = _tokenizer.Encode(output);

            // Validação mais robusta: verifica listas vazias e presença do token <unk> (ID 0)
            if (!inputTokens.Any() || !outputTokens.Any() || inputTokens.Contains(Tokenizer.UnknownTokenId) || outputTokens.Contains(Tokenizer.UnknownTokenId))
            {
                Console.WriteLine($"AVISO: Tokens inválidos ou <unk> detectado para entrada '{normalizedInput}' ou saída '{output}'.");
                Console.WriteLine($"Tokens Entrada: [{string.Join(",", inputTokens)}]");
                Console.WriteLine($"Tokens Saída: [{string.Join(",", outputTokens)}]");
                Console.WriteLine("Pulando este par...");
                continue; // Pula para o próximo par
            }

            // Log dos tokens antes de chamar o treino
            // Console.WriteLine($"Tokens Entrada (válidos): [{string.Join(",", inputTokens)}]");
            // Console.WriteLine($"Tokens Saída (válidos): [{string.Join(",", outputTokens)}]");

            // Chama o método de treino do modelo
            _model.Train(inputTokens, outputTokens);
            successfulPairs++;

            // REMOVIDO: Não salva o modelo a cada iteração
            // _model.SaveModel();
        }

        Console.WriteLine("\n--- Treinamento Concluído ---");
        Console.WriteLine($"{successfulPairs}/{trainingData.Count} pares processados com sucesso.");

        // **AJUSTE PRINCIPAL: Salva o modelo UMA VEZ após o loop de treinamento**
        if (successfulPairs > 0) // Salva apenas se treinou algo
        {
            Console.WriteLine("Salvando o modelo treinado...");
            _model.SaveModel();
        }
        else
        {
            Console.WriteLine("Nenhum par de treinamento bem-sucedido, modelo não foi salvo.");
        }
    }

    private static List<(string input, string output)> GetTrainingData()
    {
         // (Lista de pares de treinamento omitida para brevidade - igual à anterior)
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
            ("Hello", "Hello! What is on your mind?"), // Normalizado
            ("Hello", "Hey there! Nice to see you!"),
            ("Hi", "Hello! What is on your mind?"),    // Normalizado
            ("Hi", "Hi! How can I assist you?"),
            ("Good morning", "Good morning to you too!"),
            ("Hey there", "Hey! Nice to hear from you."),
            ("What is the largest planet?", "Jupiter is the largest planet in our solar system."),
            ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
            ("What is the boiling point of water?", "The boiling point of water is 100 degrees Celsius at standard pressure."),
            ("What is the tallest mountain?", "Mount Everest is the tallest mountain in the world."),
            ("Who are you?", "I am a chatbot created to answer your questions!"), // Normalizado
            ("What can you do?", "I can answer questions about capitals, general knowledge, and chat about various topics!"),
            ("What is your name?", "I am Ninfa, nice to meet you!"), // Normalizado
            ("Tell me something interesting", "Did you know that octopuses have three hearts?"),
            ("I do not know what to ask", "No worries! How about asking me about a country's capital or a fun fact?"), // Normalizado
            ("What is up?", "Not much, just hanging out in the digital world. What is up with you?"), // Normalizado
            ("Help", "I am here to help! Try asking about a capital city or a general knowledge question."), // Normalizado
            ("Tell me about France", "France is known for its culture, wine, and the Eiffel Tower in Paris."),
            ("More about France", "France has a rich history with landmarks like the Louvre and Versailles."),
            ("Tell me about Brazil", "Brazil is famous for Carnival, the Amazon rainforest, and soccer."),
            ("More about Brazil", "Brazil's capital is Brasília, and it has vibrant cities like Rio de Janeiro.")
        };
    }
}