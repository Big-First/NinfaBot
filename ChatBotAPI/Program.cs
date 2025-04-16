using System;
using System.Collections.Generic;
using System.IO;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using ChatBotAPI.Core;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;

var builder = WebApplication.CreateBuilder(args);

// Register services with dependency injection
builder.Services.AddSingleton<Model>(provider =>
{
    string tokenizerConfigPath = Path.Combine(Directory.GetCurrentDirectory(), "Vocabularys", "tokenizer.json");
    if (!File.Exists(tokenizerConfigPath))
    {
        throw new FileNotFoundException($"Tokenizer config file not found at: {tokenizerConfigPath}");
    }
    string json = File.ReadAllText(tokenizerConfigPath);
    return JsonSerializer.Deserialize<Model>(json) ?? throw new JsonException("Failed to deserialize model config");
});
builder.Services.AddSingleton<NeuralModel>(provider =>
{
    int embeddingSize = 128;
    int maxSequenceLength = 50;
    var modelConfig = provider.GetRequiredService<Model>();
    return new BinaryTreeNeuralModel(modelConfig, embeddingSize, maxSequenceLength);
});
builder.Services.AddSingleton<Tokenizer>(provider =>
{
    int vocabSize = 10000;
    int maxSequenceLength = 50;
    string tokenizerConfigPath = Path.Combine(Directory.GetCurrentDirectory(), "Vocabularys", "tokenizer.json");
    if (!File.Exists(tokenizerConfigPath))
    {
        throw new FileNotFoundException($"Tokenizer config file not found at: {tokenizerConfigPath}");
    }
    return new Tokenizer(tokenizerConfigPath, maxSequenceLength, vocabSize);
});
builder.Services.AddSingleton<ChatBotService>();
builder.Services.AddSingleton<Trainer>(provider =>
{
    int maxDepth = 5;
    var model = provider.GetRequiredService<NeuralModel>();
    var tokenizer = provider.GetRequiredService<Tokenizer>();
    return new Trainer(model, tokenizer, maxDepth);
});

var app = builder.Build();

app.UseWebSockets();

// Train the model after services are built
using (var scope = app.Services.CreateScope())
{
    var trainer = scope.ServiceProvider.GetRequiredService<Trainer>();
    List<string> trainingData = new List<string>
    {
        "hello how are you",
        "what is the weather like",
        "tell me a joke"
    };
    trainer.Train(trainingData, epochs: 10);
}

app.Map("/chat", async context =>
{
    if (context.WebSockets.IsWebSocketRequest)
    {
        using var webSocket = await context.WebSockets.AcceptWebSocketAsync();
        var chatService = context.RequestServices.GetRequiredService<ChatBotService>();
        await HandleWebSocketAsync(webSocket, chatService, context.RequestAborted);
    }
    else
    {
        context.Response.StatusCode = StatusCodes.Status400BadRequest;
    }
});

await app.RunAsync();

async Task HandleWebSocketAsync(WebSocket webSocket, ChatBotService chatService, CancellationToken cancellationToken)
{
    var buffer = new byte[1024 * 4];
    while (webSocket.State == WebSocketState.Open)
    {
        var result = await webSocket.ReceiveAsync(new ArraySegment<byte>(buffer), cancellationToken);
        if (result.MessageType == WebSocketMessageType.Text)
        {
            string input = Encoding.UTF8.GetString(buffer, 0, result.Count);
            Console.WriteLine($"Received: {input}");
            await chatService.ProcessMessage(webSocket, input);
        }
        else if (result.MessageType == WebSocketMessageType.Close)
        {
            await webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Closing", cancellationToken);
        }
    }
}