using System.Net.WebSockets;
using System.Text;
using System.Text.Json; // Usado para JsonException, se necessário
using ChatBotAPI.Core;
using ChatBotAPI.Settings; // Assumindo que ModelSettings, Model, TokenWrapper, etc. estão aqui
// using ChatBotAPI.Settings; // Remova se ModelSettings estiver em Core
using Microsoft.Extensions.Options;
using Microsoft.AspNetCore.Builder; // Necessário para WebApplication, IApplicationBuilder
using Microsoft.AspNetCore.Http; // Necessário para HttpContext, StatusCodes
using Microsoft.Extensions.DependencyInjection; // Necessário para IServiceCollection, IServiceProvider
using Microsoft.Extensions.Hosting; // Necessário para WebApplication.CreateBuilder

var builder = WebApplication.CreateBuilder(args);

// *** 1. Configuração ***
builder.Services.Configure<ModelSettings>(builder.Configuration.GetSection("ModelSettings"));
builder.Services.AddSingleton(resolver => resolver.GetRequiredService<IOptions<ModelSettings>>().Value);

// *** 2. Registro de Serviços com DI ***

// Registra o Model (lido do JSON via TokenWrapper) como Singleton
builder.Services.AddSingleton<Model>(provider =>
{
    var settings = provider.GetRequiredService<ModelSettings>();
    string tokenizerConfigPath = Path.GetFullPath(settings.TokenizerConfigPath);

    Console.WriteLine($"Loading tokenizer config from: {tokenizerConfigPath}");

    if (!File.Exists(tokenizerConfigPath))
    {
        throw new FileNotFoundException($"Tokenizer config file not found at the specified path: {tokenizerConfigPath}");
    }
    try
    {
        string json = File.ReadAllText(tokenizerConfigPath);
        var options = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true // Mantém isso para robustez na desserialização
        };

        var tokenWrapper = JsonSerializer.Deserialize<TokenWrapper>(json, options);

        if (tokenWrapper == null) {
             throw new JsonException($"Failed to deserialize tokenizer config file to TokenWrapper object. Path: {tokenizerConfigPath}");
        }
         // *** CORRIGIDO: Usa PascalCase para acessar propriedades C# ***
        if (tokenWrapper.model == null) {
             throw new JsonException($"'Model' field is missing or null inside the TokenWrapper object after deserialization. Path: {tokenizerConfigPath}");
        }
        // *** CORRIGIDO: Usa PascalCase para acessar propriedades C# ***
        if (tokenWrapper.model.vocab == null) {
             throw new JsonException($"'Vocab' field is missing or null in the nested Model object. Path: {tokenizerConfigPath}");
        }

        // *** CORRIGIDO: Usa PascalCase para acessar propriedades C# ***
        Console.WriteLine($"TokenWrapper loaded successfully. Version: {tokenWrapper.version}. Nested Model Type: {tokenWrapper.model.type}. Vocabulary size: {tokenWrapper.model.vocab.Count}");

        // *** CORRIGIDO: Usa PascalCase para acessar propriedades C# ***
        return tokenWrapper.model; // Retorna o objeto Model de dentro do Wrapper
    }
    catch (JsonException ex) {
         Console.Error.WriteLine($"JSON Deserialization Error for {tokenizerConfigPath}: {ex.Message} Path: {ex.Path}, Line: {ex.LineNumber}, Pos: {ex.BytePositionInLine}");
        throw new JsonException($"Failed to deserialize tokenizer config file. Check JSON format and content. Path: {tokenizerConfigPath}", ex);
    }
    catch (Exception ex) {
        Console.Error.WriteLine($"Error reading tokenizer config file {tokenizerConfigPath}: {ex.Message}");
        throw;
    }
});

builder.Services.AddSingleton<Tokenizer>(provider =>
{
    var settings = provider.GetRequiredService<ModelSettings>();
    var loadedModel = provider.GetRequiredService<Model>();

    // *** CORRIGIDO: Usa PascalCase para acessar propriedades C# ***
    return new Tokenizer(
        loadedModel.vocab ?? new Dictionary<string, int>(), // Usa PascalCase (Vocab)
        settings.MaxSequenceLength,
        settings.VocabSizeLimit
    );
});

// Corrigido: Registra NeuralModel como a implementação concreta
// Se precisar da interface, registre como: builder.Services.AddSingleton<NeuralModel, BinaryTreeNeuralModel>(provider => ...);
builder.Services.AddSingleton<BinaryTreeNeuralModel>(provider => // Registra o tipo concreto
{
    var settings = provider.GetRequiredService<ModelSettings>();
    // var loadedModel = provider.GetRequiredService<Model>(); // Não precisa mais do Model aqui
    var tokenizer = provider.GetRequiredService<Tokenizer>();
    int actualVocabSizeUsedByTokenizer = tokenizer.ActualVocabSize;

    Console.WriteLine($"Initializing Neural Model. Embedding Size: {settings.EmbeddingSize}, Sequence Length: {settings.MaxSequenceLength}, Actual Vocab Size: {actualVocabSizeUsedByTokenizer}");

    // *** CORRIGIDO: Usa a variável com o tamanho real do vocabulário ***
    var neuralModel = new BinaryTreeNeuralModel(
        actualVocabSizeUsedByTokenizer, // Passa o tamanho real calculado
        settings.EmbeddingSize,
        settings.MaxSequenceLength);

    return neuralModel;
});

// Se outros serviços dependem da *interface* NeuralModel, adicione um registro que a aponte para a implementação:
builder.Services.AddSingleton<NeuralModel>(provider => provider.GetRequiredService<BinaryTreeNeuralModel>());


builder.Services.AddSingleton<ChatBotService>(); // Depende de NeuralModel (interface) e Tokenizer
builder.Services.AddSingleton<Trainer>(provider => { // Depende de NeuralModel (interface) e Tokenizer
    var settings = provider.GetRequiredService<ModelSettings>();
    var model = provider.GetRequiredService<NeuralModel>(); // Resolve a interface
    var tokenizer = provider.GetRequiredService<Tokenizer>();
    return new Trainer(model, tokenizer, settings.MaxTreeDepth);
});


var app = builder.Build();

// Configuração do Pipeline de Requisição HTTP
app.UseWebSockets();

// Código de Treinamento
using (var scope = app.Services.CreateScope())
{
    var trainer = scope.ServiceProvider.GetRequiredService<Trainer>();
    var settings = scope.ServiceProvider.GetRequiredService<ModelSettings>();
    List<string> trainingData = new List<string> { /* ... dados ... */ };
    if (trainingData.Any()) {
        Console.WriteLine($"Starting training with {trainingData.Count} samples for {settings.TrainingEpochs} epochs...");
        trainer.Train(trainingData, epochs: settings.TrainingEpochs);
        Console.WriteLine("Training finished.");
    } else {
        Console.WriteLine("No training data provided, skipping training.");
    }
}

// Mapeamento do Endpoint WebSocket
app.Map("/chat", async context =>
{
    if (context.WebSockets.IsWebSocketRequest)
    {
        Console.WriteLine("WebSocket request received for /chat. Accepting connection...");
        try
        {
            using var webSocket = await context.WebSockets.AcceptWebSocketAsync();
            // Resolve ChatBotService para esta requisição
            var chatService = context.RequestServices.GetRequiredService<ChatBotService>();
            await HandleWebSocketAsync(webSocket, chatService, context.RequestAborted);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error accepting WebSocket connection: {ex.Message}");
            if (!context.Response.HasStarted) {
                 context.Response.StatusCode = StatusCodes.Status500InternalServerError;
            }
        }
    }
    else
    {
        Console.WriteLine("Received non-WebSocket request for /chat. Responding with 400 Bad Request.");
        context.Response.StatusCode = StatusCodes.Status400BadRequest;
        await context.Response.WriteAsync("This endpoint requires a WebSocket connection.");
    }
});

// Definição do Handler WebSocket
async Task HandleWebSocketAsync(WebSocket webSocket, ChatBotService chatService, CancellationToken cancellationToken)
{
    var buffer = new byte[1024 * 4];
    Console.WriteLine($"WebSocket connection {webSocket.GetHashCode()} established.");
    try
    {
        while (webSocket.State == WebSocketState.Open && !cancellationToken.IsCancellationRequested)
        {
            var result = await webSocket.ReceiveAsync(new ArraySegment<byte>(buffer), cancellationToken);

            if (result.MessageType == WebSocketMessageType.Text)
            {
                if (result.Count == 0) continue;
                string input = Encoding.UTF8.GetString(buffer, 0, result.Count);
                Console.WriteLine($"Connection {webSocket.GetHashCode()}: Received: {input}");

                // Chama o serviço UMA VEZ
                await chatService.ProcessMessage(webSocket, input);
                Console.WriteLine($"Connection {webSocket.GetHashCode()}: Response potentially sent for: {input}");
            }
            else if (result.MessageType == WebSocketMessageType.Close)
            {
                Console.WriteLine($"Connection {webSocket.GetHashCode()}: WebSocket closing request received.");
                await webSocket.CloseAsync(result.CloseStatus ?? WebSocketCloseStatus.NormalClosure, result.CloseStatusDescription, CancellationToken.None);
                break;
            }
        }
    }
    catch (WebSocketException ex) when (ex.WebSocketErrorCode == WebSocketError.ConnectionClosedPrematurely) {
        Console.WriteLine($"Connection {webSocket.GetHashCode()}: WebSocket connection closed prematurely.");
    }
    catch (OperationCanceledException) {
        Console.WriteLine($"Connection {webSocket.GetHashCode()}: WebSocket operation cancelled.");
        if (webSocket.State == WebSocketState.Open || webSocket.State == WebSocketState.CloseReceived) {
            await webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Operation cancelled", CancellationToken.None);
        }
    }
    catch (Exception ex) {
        Console.Error.WriteLine($"Connection {webSocket.GetHashCode()}: Error in WebSocket handling: {ex.ToString()}");
        if (webSocket.State == WebSocketState.Open || webSocket.State == WebSocketState.CloseReceived) {
            await webSocket.CloseAsync(WebSocketCloseStatus.InternalServerError, "Unexpected server error during communication", CancellationToken.None);
        }
    }
    finally {
        Console.WriteLine($"WebSocket connection {webSocket.GetHashCode()} processing finished. Final state: {webSocket.State}");
    }
}

Console.WriteLine("Setup complete. Starting the web server...");
await app.RunAsync(); // Inicia o servidor