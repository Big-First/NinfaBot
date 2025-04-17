using System.Net.WebSockets;
using System.Text;
using System.Text.Json; // Usado para JsonException, se necessário
using ChatBotAPI.Core; // Namespace principal para suas classes
// Remova 'using ChatBotAPI.Settings;' se ModelSettings está em Core
using Microsoft.Extensions.Options;
// Para List<>
// Para Path, File
using ChatBotAPI.Settings; // Para Linq (Any, Select)

var builder = WebApplication.CreateBuilder(args);

// *** 1. Configuração ***
builder.Services.Configure<ModelSettings>(builder.Configuration.GetSection("ModelSettings"));
builder.Services.AddSingleton(resolver => resolver.GetRequiredService<IOptions<ModelSettings>>().Value);

// *** 2. Registro de Serviços com DI ***

// Model (lido do JSON via TokenWrapper)
builder.Services.AddSingleton<Model>(provider =>
{
    var settings = provider.GetRequiredService<ModelSettings>();
    string tokenizerConfigPath = Path.GetFullPath(settings.TokenizerConfigPath);
    Console.WriteLine($"Loading tokenizer config from: {tokenizerConfigPath}");
    if (!File.Exists(tokenizerConfigPath)) throw new FileNotFoundException($"Tokenizer config file not found: {tokenizerConfigPath}");

    try
    {
        string json = File.ReadAllText(tokenizerConfigPath);
        // Opção PropertyNameCaseInsensitive ainda é útil se o JSON *puder* variar o case
        var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
        var tokenWrapper = JsonSerializer.Deserialize<TokenWrapper>(json, options); // Desserializa para TokenWrapper
        if (tokenWrapper != null) {
            Console.WriteLine($"DEBUG: tokenWrapper.version = {tokenWrapper.version ?? "null"}");
            Console.WriteLine($"DEBUG: tokenWrapper.model is null? {tokenWrapper.model == null}");
            if (tokenWrapper.model != null) {
                Console.WriteLine($"DEBUG: tokenWrapper.model.type = {tokenWrapper.model.type ?? "null"}");
                Console.WriteLine($"DEBUG: tokenWrapper.model.vocab is null? {tokenWrapper.model.vocab == null}"); // *** O PONTO CRÍTICO ***
                if (tokenWrapper.model.vocab != null) {
                    Console.WriteLine($"DEBUG: tokenWrapper.model.vocab Count = {tokenWrapper.model.vocab.Count}");
                } else {
                    Console.WriteLine("DEBUG: tokenWrapper.model.vocab IS NULL after deserialization!");
                }
            } else {
                Console.WriteLine("DEBUG: tokenWrapper.model IS NULL after deserialization!");
            }
        } else {
            Console.WriteLine("DEBUG: tokenWrapper IS NULL after deserialization!");
        }

        // *** CORREÇÃO: Usa lowercase para acessar propriedades C# ***
        Console.WriteLine($"TokenWrapper loaded. Version: {tokenWrapper.version}. Model Type: {tokenWrapper.model.type}. Vocab size: {tokenWrapper.model.vocab.Count}");

        // *** CORREÇÃO: Retorna a propriedade correta (lowercase) ***
        return tokenWrapper.model; // Retorna o objeto Model aninhado
    }
    catch (Exception ex) when (ex is JsonException || ex is NotSupportedException)
    { /* ... log erro ... */ throw; }
    catch (Exception ex)
    { /* ... log erro ... */ throw; }
});

// Tokenizer
// *** Verifique também o registro do Tokenizer ***
builder.Services.AddSingleton<Tokenizer>(provider =>
{
    var settings = provider.GetRequiredService<ModelSettings>();
    var loadedModel = provider.GetRequiredService<Model>();
    // *** CORREÇÃO: Usa lowercase para acessar a propriedade C# ***
    // Garante que a propriedade 'vocab' (lowercase) de loadedModel não seja null
    return new Tokenizer(loadedModel.vocab ?? new Dictionary<string, int>(), settings.MaxSequenceLength, settings.VocabSizeLimit);
});

// NeuralModel (implementação concreta)
builder.Services.AddSingleton<TorchSharpModel>(provider => // Registra o tipo concreto
{
    var settings = provider.GetRequiredService<ModelSettings>();
    var tokenizer = provider.GetRequiredService<Tokenizer>();
    int actualVocabSize = tokenizer.ActualVocabSize;
    int paddingIdx = tokenizer.PadTokenId; // Obtém o índice de padding

    Console.WriteLine($"Initializing TorchSharp Model. Vocab Size: {actualVocabSize}, Embedding Size: {settings.EmbeddingSize}, Padding Idx: {paddingIdx}");

    // Passa os parâmetros necessários para o modelo TorchSharp
    var torchModel = new TorchSharpModel(
        actualVocabSize,
        settings.EmbeddingSize,
        paddingIdx); // Passa o índice de padding

    return torchModel;
});

// ChatBotService
builder.Services.AddSingleton<ChatBotService>(provider => {
    var model = provider.GetRequiredService<TorchSharpModel>(); // Pede o modelo TorchSharp
    var tokenizer = provider.GetRequiredService<Tokenizer>();
    return new ChatBotService(model, tokenizer);
});

// Trainer
builder.Services.AddSingleton<Trainer>(provider => {
    var model = provider.GetRequiredService<TorchSharpModel>();
    // Coloque um breakpoint AQUI
    var tokenizer = provider.GetRequiredService<Tokenizer>(); // Pede o Tokenizer
    Console.WriteLine($"DEBUG: Program.cs - Injecting Tokenizer into Trainer. Is null? {tokenizer == null}");// Para learning rate, se necessário
    // Aqui você pode obter a taxa de aprendizado das configurações (settings.LearningRate por exemplo)
    double learningRate = 0.001; // Ou use settings.LearningRate
    return new Trainer(model, tokenizer, learningRate);
});


// *** Construção do App ***
var app = builder.Build();

// *** Configuração do Pipeline de Requisição HTTP ***
app.UseWebSockets(); // Essencial para WebSockets

// *** TREINAMENTO NA INICIALIZAÇÃO (INCORPORADO) ***
Console.WriteLine("--- Starting Training Phase ---");
using (var scope = app.Services.CreateScope()) // Cria um escopo para resolver serviços
{
    var trainer = scope.ServiceProvider.GetRequiredService<Trainer>();
    var settings = scope.ServiceProvider.GetRequiredService<ModelSettings>(); // Obtém configurações (ex: número de épocas)

    // 1. Obter os dados brutos (input, output) da função estática
    List<(string input, string output)> rawTrainingData = GetTrainingData(); // Chama a função definida abaixo

    // 2. Formatar os dados como sequências únicas para next-word prediction
    List<string> trainingSequences = rawTrainingData
        .Select(pair => $"{pair.input} {pair.output}") // Concatena input e output
        .ToList();

    // 3. Chamar o Trainer com as sequências formatadas
    if (trainingSequences.Any())
    {
        Console.WriteLine($"Starting training with {trainingSequences.Count} combined sequences for {settings.TrainingEpochs} epochs...");
        // O Trainer.cs ajustado processa List<string> com next-word prediction
        trainer.Train(trainingSequences, epochs: settings.TrainingEpochs);
        Console.WriteLine("--- Training Finished ---");
    }
    else
    {
        Console.WriteLine("No training data provided/generated, skipping training.");
        Console.WriteLine("--- Training Phase Skipped ---");
    }
} // Fim do escopo de serviço para treinamento


// *** Mapeamento de Endpoints ***
app.Map("/chat", async context =>
{
    if (context.WebSockets.IsWebSocketRequest)
    {
        Console.WriteLine("WebSocket request received for /chat. Accepting connection...");
        try
        {
            using var webSocket = await context.WebSockets.AcceptWebSocketAsync();
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
async Task HandleWebSocketAsync(WebSocket webSocket, ChatBotService chatService, CancellationToken cancellationToken)
{
    var buffer = new byte[1024 * 4];
    Console.WriteLine($"WebSocket connection {webSocket.GetHashCode()} established.");
    try { /* ... (código do handler como antes) ... */
        while (webSocket.State == WebSocketState.Open && !cancellationToken.IsCancellationRequested)
        {
            var result = await webSocket.ReceiveAsync(new ArraySegment<byte>(buffer), cancellationToken);

            if (result.MessageType == WebSocketMessageType.Text) {
                if (result.Count == 0) continue;
                string input = Encoding.UTF8.GetString(buffer, 0, result.Count);
                Console.WriteLine($"Connection {webSocket.GetHashCode()}: Received: {input}");
                await chatService.ProcessMessage(webSocket, input);
                Console.WriteLine($"Connection {webSocket.GetHashCode()}: Response potentially sent for: {input}");
            }
            else if (result.MessageType == WebSocketMessageType.Close) {
                Console.WriteLine($"Connection {webSocket.GetHashCode()}: WebSocket closing request received.");
                await webSocket.CloseAsync(result.CloseStatus ?? WebSocketCloseStatus.NormalClosure, result.CloseStatusDescription, CancellationToken.None);
                break;
            }
        }
    }
     catch (WebSocketException ex) when (ex.WebSocketErrorCode == WebSocketError.ConnectionClosedPrematurely) { Console.WriteLine($"Connection {webSocket.GetHashCode()}: WebSocket connection closed prematurely."); }
     catch (OperationCanceledException) { Console.WriteLine($"Connection {webSocket.GetHashCode()}: WebSocket operation cancelled."); /* ... (fechar socket) ... */ }
     catch (Exception ex) { Console.Error.WriteLine($"Connection {webSocket.GetHashCode()}: Error in WebSocket handling: {ex.ToString()}"); /* ... (fechar socket) ... */ }
     finally { Console.WriteLine($"WebSocket connection {webSocket.GetHashCode()} processing finished. Final state: {webSocket.State}"); }
}

// Função estática para fornecer os dados de treinamento
static List<(string input, string output)> GetTrainingData()
{
    // Cole aqui a sua lista de pares (input, output)
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
        ("Hello", "Hello! What is on your mind?"),
        ("Hello", "Hey there! Nice to see you!"),
        ("Hi", "Hello! What is on your mind?"),
        ("Hi", "Hi! How can I assist you?"),
        ("Good morning", "Good morning to you too!"),
        ("Hey there", "Hey! Nice to hear from you."),
        ("What is the largest planet?", "Jupiter is the largest planet in our solar system."),
        ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
        ("What is the boiling point of water?", "The boiling point of water is 100 degrees Celsius at standard pressure."),
        ("What is the tallest mountain?", "Mount Everest is the tallest mountain in the world."),
        ("Who are you?", "I am a chatbot created to answer your questions!"),
        ("What can you do?", "I can answer questions about capitals, general knowledge, and chat about various topics!"),
        ("What is your name?", "I am Ninfa, nice to meet you!"),
        ("Tell me something interesting", "Did you know that octopuses have three hearts?"),
        ("I do not know what to ask", "No worries! How about asking me about a country's capital or a fun fact?"),
        ("What is up?", "Not much, just hanging out in the digital world. What is up with you?"),
        ("Help", "I am here to help! Try asking about a capital city or a general knowledge question."),
        ("Tell me about France", "France is known for its culture, wine, and the Eiffel Tower in Paris."),
        ("More about France", "France has a rich history with landmarks like the Louvre and Versailles."),
        ("Tell me about Brazil", "Brazil is famous for Carnival, the Amazon rainforest, and soccer."),
        ("More about Brazil", "Brazil's capital is Brasília, and it has vibrant cities like Rio de Janeiro.")
    };
}


// *** Inicialização Final ***
// ... (Mapeamento de /chat e HandleWebSocketAsync como antes) ...
// ... (Função GetTrainingData como antes) ...

Console.WriteLine("Setup complete. Starting the web server...");
await app.RunAsync();// Mantém o servidor rodando