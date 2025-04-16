using System.Net.WebSockets;
using System.Text;
using ChatBotAPI.Core;

var builder = WebApplication.CreateBuilder(args);

// Register services with dependency injection
builder.Services.AddSingleton<Model>(provider => 
{
    int vocabSize = 10000;
    int embeddingSize = 128;
    int maxSequenceLength = 50;
    var model = new BinaryTreeNeuralModel(vocabSize, embeddingSize, maxSequenceLength);
    return model;
});
builder.Services.AddSingleton<Tokenizer>(provider =>
{
    int vocabSize = 10000;
    int maxSequenceLength = 50;
    string tokenizerConfigPath = System.IO.Path.Combine(System.IO.Directory.GetCurrentDirectory(), "Vocabularys", "tokenizer.json");
    return new Tokenizer(tokenizerConfigPath, maxSequenceLength, vocabSize);
});
builder.Services.AddSingleton<ChatBotService>();
builder.Services.AddSingleton<Trainer>(provider =>
{
    int maxDepth = 5;
    var model = provider.GetRequiredService<Model>();
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
        context.Response.StatusCode = 400;
    }
});

app.Run();

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