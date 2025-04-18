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
    if (!File.Exists(tokenizerConfigPath))
        throw new FileNotFoundException($"Tokenizer config file not found: {tokenizerConfigPath}");

    try
    {
        string json = File.ReadAllText(tokenizerConfigPath);
        // Opção PropertyNameCaseInsensitive ainda é útil se o JSON *puder* variar o case
        var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
        var tokenWrapper = JsonSerializer.Deserialize<TokenWrapper>(json, options); // Desserializa para TokenWrapper
        if (tokenWrapper != null)
        {
            Console.WriteLine($"DEBUG: tokenWrapper.version = {tokenWrapper.version ?? "null"}");
            Console.WriteLine($"DEBUG: tokenWrapper.model is null? {tokenWrapper.model == null}");
            if (tokenWrapper.model != null)
            {
                Console.WriteLine($"DEBUG: tokenWrapper.model.type = {tokenWrapper.model.type ?? "null"}");
                Console.WriteLine(
                    $"DEBUG: tokenWrapper.model.vocab is null? {tokenWrapper.model.vocab == null}"); // *** O PONTO CRÍTICO ***
                if (tokenWrapper.model.vocab != null)
                {
                    Console.WriteLine($"DEBUG: tokenWrapper.model.vocab Count = {tokenWrapper.model.vocab.Count}");
                }
                else
                {
                    Console.WriteLine("DEBUG: tokenWrapper.model.vocab IS NULL after deserialization!");
                }
            }
            else
            {
                Console.WriteLine("DEBUG: tokenWrapper.model IS NULL after deserialization!");
            }
        }
        else
        {
            Console.WriteLine("DEBUG: tokenWrapper IS NULL after deserialization!");
        }

        // *** CORREÇÃO: Usa lowercase para acessar propriedades C# ***
        Console.WriteLine(
            $"TokenWrapper loaded. Version: {tokenWrapper.version}. Model Type: {tokenWrapper.model.type}. Vocab size: {tokenWrapper.model.vocab.Count}");

        // *** CORREÇÃO: Retorna a propriedade correta (lowercase) ***
        return tokenWrapper.model; // Retorna o objeto Model aninhado
    }
    catch (Exception ex) when (ex is JsonException || ex is NotSupportedException)
    {
        /* ... log erro ... */
        throw;
    }
    catch (Exception ex)
    {
        /* ... log erro ... */
        throw;
    }
});

// Tokenizer
// *** Verifique também o registro do Tokenizer ***
builder.Services.AddSingleton<Tokenizer>(provider =>
{
    var settings = provider.GetRequiredService<ModelSettings>();
    var loadedModel = provider.GetRequiredService<Model>();
    // *** CORREÇÃO: Usa lowercase para acessar a propriedade C# ***
    // Garante que a propriedade 'vocab' (lowercase) de loadedModel não seja null
    return new Tokenizer(loadedModel.vocab ?? new Dictionary<string, int>(), settings.MaxSequenceLength,
        settings.VocabSizeLimit);
});

// NeuralModel (implementação concreta)
builder.Services.AddSingleton<TorchSharpModel>(provider => // Registra o tipo concreto
{
    var settings = provider.GetRequiredService<ModelSettings>();
    var tokenizer = provider.GetRequiredService<Tokenizer>();
    int actualVocabSize = tokenizer.ActualVocabSize;
    int paddingIdx = tokenizer.PadTokenId; // Obtém o índice de padding

    Console.WriteLine(
        $"Initializing TorchSharp Model. Vocab Size: {actualVocabSize}, Embedding Size: {settings.EmbeddingSize}, Padding Idx: {paddingIdx}");

    // Passa os parâmetros necessários para o modelo TorchSharp
    var torchModel = new TorchSharpModel(
        actualVocabSize,
        settings.EmbeddingSize,
        paddingIdx); // Passa o índice de padding

    return torchModel;
});

// ChatBotService
builder.Services.AddSingleton<ChatBotService>(provider =>
{
    var model = provider.GetRequiredService<TorchSharpModel>(); // Pede o modelo TorchSharp
    var tokenizer = provider.GetRequiredService<Tokenizer>();
    return new ChatBotService(model, tokenizer);
});

// Trainer
builder.Services.AddSingleton<Trainer>(provider =>
{
    var model = provider.GetRequiredService<TorchSharpModel>();
    // Coloque um breakpoint AQUI
    var tokenizer = provider.GetRequiredService<Tokenizer>(); // Pede o Tokenizer
    Console.WriteLine(
        $"DEBUG: Program.cs - Injecting Tokenizer into Trainer. Is null? {tokenizer == null}"); // Para learning rate, se necessário
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
    var settings =
        scope.ServiceProvider.GetRequiredService<ModelSettings>(); // Obtém configurações (ex: número de épocas)

    // 1. Obter os dados brutos (input, output) da função estática
    int numberOfTrainingPairs = 1500; // Ou leia de settings, ex: settings.NumberOfTrainingPairs
    Console.WriteLine($"Generating {numberOfTrainingPairs} training pairs using TrainingDataGenerator...");
    // Chama o método estático da classe que você criou (ajuste o nome da classe se necessário)
    List<(string input, string output)> rawTrainingData = TrainingDataGenerator.GetTrainingData(numberOfTrainingPairs);
    Console.WriteLine($"Generated {rawTrainingData.Count} actual pairs.");// Chama a função definida abaixo

    // 2. Formatar os dados como sequências únicas para next-word prediction
    List<string> trainingSequences = rawTrainingData
        .Select(pair => $"{pair.input} {pair.output}") // Concatena input e output
        .ToList();

    // 3. Chamar o Trainer com as sequências formatadas
    if (trainingSequences.Any())
    {
        Console.WriteLine(
            $"Starting training with {trainingSequences.Count} combined sequences for {settings.TrainingEpochs} epochs...");
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
            if (!context.Response.HasStarted)
            {
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
    try
    {
        /* ... (código do handler como antes) ... */
        while (webSocket.State == WebSocketState.Open && !cancellationToken.IsCancellationRequested)
        {
            var result = await webSocket.ReceiveAsync(new ArraySegment<byte>(buffer), cancellationToken);

            if (result.MessageType == WebSocketMessageType.Text)
            {
                if (result.Count == 0) continue;
                string input = Encoding.UTF8.GetString(buffer, 0, result.Count);
                Console.WriteLine($"Connection {webSocket.GetHashCode()}: Received: {input}");
                await chatService.ProcessMessage(webSocket, input);
                Console.WriteLine($"Connection {webSocket.GetHashCode()}: Response potentially sent for: {input}");
            }
            else if (result.MessageType == WebSocketMessageType.Close)
            {
                Console.WriteLine($"Connection {webSocket.GetHashCode()}: WebSocket closing request received.");
                await webSocket.CloseAsync(result.CloseStatus ?? WebSocketCloseStatus.NormalClosure,
                    result.CloseStatusDescription, CancellationToken.None);
                break;
            }
        }
    }
    catch (WebSocketException ex) when (ex.WebSocketErrorCode == WebSocketError.ConnectionClosedPrematurely)
    {
        Console.WriteLine($"Connection {webSocket.GetHashCode()}: WebSocket connection closed prematurely.");
    }
    catch (OperationCanceledException)
    {
        Console.WriteLine(
            $"Connection {webSocket.GetHashCode()}: WebSocket operation cancelled."); /* ... (fechar socket) ... */
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine(
            $"Connection {webSocket.GetHashCode()}: Error in WebSocket handling: {ex.ToString()}"); /* ... (fechar socket) ... */
    }
    finally
    {
        Console.WriteLine(
            $"WebSocket connection {webSocket.GetHashCode()} processing finished. Final state: {webSocket.State}");
    }
}

/*
// Função estática para fornecer os dados de treinamento
static List<(string input, string output)> GetTrainingData()
{
    // Cole aqui a sua lista de pares (input, output)
    return new List<(string input, string output)>
    {
        ("Hello", "Hi! How can I help you today?"),
        ("Hello", "Hello! What is on your mind?"),
        ("Hello", "Hey there! Nice to see you!"),
        ("Hi", "Hello! What is on your mind?"),
        ("Hi", "Hi! How can I assist you?"),
        ("Hey", "Hey! What’s up?"),
        ("Good morning", "Good morning to you too!"),
        ("Good afternoon", "Good afternoon! Hope you're doing well."),
        ("Good evening", "Good evening! How was your day?"),
        ("Hey there", "Hey! Nice to hear from you."),
        ("Greetings", "Greetings! How can I assist you?"),
        ("Bye", "Goodbye! Have a nice day."),
        ("See you later", "See you!"),
        ("Good night", "Good night! Sleep well."),
        ("Farewell", "Farewell! Until next time."),
        ("Talk to you later", "Sure! I'll be here."),
        ("Thank you", "You're welcome!"),
        ("Thanks", "No problem! Happy to help."),
        ("Thanks a lot", "Anytime!"),
        ("Much appreciated", "Glad to be of help."),
        ("I appreciate it", "You're very welcome!"),
        ("Help", "I am here to help! Try asking about a capital city or a general knowledge question."),
        ("I need help", "Sure! What would you like to know?"),
        ("Can you help me?", "Absolutely! What’s your question?"),
        ("What can I ask?", "You can ask about countries, science facts, jokes, or general knowledge!"),
        ("Who are you?", "I am a chatbot created to answer your questions!"),
        ("What is your name?", "I am Ninfa, nice to meet you!"),
        ("What can you do?", "I can answer questions about capitals, general knowledge, and chat about various topics!"),
        ("What is your purpose?", "I am here to chat and answer your questions."),
        ("How old are you?", "I don't have an age, I am a computer program."),
        ("Where do you live?", "I live in the cloud, on servers!"),
        ("The capital of France is", "Paris."),
        ("The capital of Brazil is", "Brasília."),
        ("The capital of Japan is", "Tokyo."),
        ("The capital of Australia is", "Canberra."),
        ("The capital of Canada is", "Ottawa."),
        ("The capital of Germany is", "Berlin."),
        ("The capital of Italy is", "Rome."),
        ("The capital of Argentina is", "Buenos Aires."),
        ("The capital of Spain is", "Madrid."),
        ("The capital of the United States is", "Washington, D.C."),
        ("The capital of Russia is", "Moscow."),
        ("The capital of China is", "Beijing."),
        ("The capital of India is", "New Delhi."),
        ("The capital of South Africa is", "Pretoria."),
        ("The capital of Egypt is", "Cairo."),
        ("The capital of Mexico is", "Mexico City."),
        ("The capital of South Korea is", "Seoul."),
        ("The capital of the UK is", "London."),
        ("The capital of Portugal is", "Lisbon."),
        ("The capital of Peru is", "Lima."),
        ("The capital of Chile is", "Santiago."),
        ("The capital of Colombia is", "Bogotá."),
        ("The capital of Thailand is", "Bangkok."),
        ("The capital of Turkey is", "Ankara."),
        ("The capital of Indonesia is", "Jakarta."),
        ("Tell me something interesting", "Did you know that octopuses have three hearts?"),
        ("Tell me a fun fact", "Bananas are berries, but strawberries aren't!"),
        ("Did you know?", "Honey never spoils. Archaeologists have found edible honey in ancient tombs!"),
        ("What is the largest planet?", "Jupiter is the largest planet in our solar system."),
        ("What is the smallest planet?", "Mercury is the smallest planet in our solar system."),
        ("What is the hottest planet?", "Venus is the hottest due to its thick atmosphere."),
        ("What is the coldest planet?", "Neptune is the coldest in our solar system."),
        ("What is the boiling point of water?", "The boiling point of water is 100 degrees Celsius at standard pressure."),
        ("What is the freezing point of water?", "0 degrees Celsius."),
        ("What is the tallest mountain?", "Mount Everest is the tallest mountain in the world."),
        ("What is the deepest ocean?", "The Pacific Ocean is the deepest."),
        ("What is the longest river?", "The Nile River is often considered the longest."),
        ("How many continents are there?", "There are seven continents."),
        ("How many oceans are there?", "There are five oceans on Earth."),
        ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci."),
        ("Who discovered gravity?", "Isaac Newton."),
        ("What is the speed of light?", "About 299,792 kilometers per second."),
        ("Can you tell me a joke?", "Why don't scientists trust atoms? Because they make up everything!"),
        ("Tell me another joke", "Why did the bicycle fall over? It was two-tired."),
        ("Make me laugh", "I would tell you a UDP joke, but you might not get it."),
        ("Do you know a joke?", "Why did the scarecrow win an award? Because he was outstanding in his field!"),
        ("What is up?", "Not much, just hanging out in the digital world. What is up with you?"),
        ("How are you?", "I’m just a bunch of code, but I’m functioning perfectly!"),
        ("What are you doing?", "Chatting with you!"),
        ("I do not know what to ask", "No worries! How about asking me about a country's capital or a fun fact?"),
        ("I'm bored", "Want to hear a joke or learn a fun fact?"),
        ("Do you have emotions?", "Not really, but I can pretend!"),
        ("Are you real?", "As real as a line of code can be!"),
        ("Do you dream?", "Only in binary."),
        ("Tell me about France", "France is known for its culture, wine, and the Eiffel Tower in Paris."),
        ("More about France", "France has a rich history with landmarks like the Louvre and Versailles."),
        ("Tell me about Brazil", "Brazil is famous for Carnival, the Amazon rainforest, and soccer."),
        ("More about Brazil", "Brazil's capital is Brasília, and it has vibrant cities like Rio de Janeiro."),
        ("Tell me about Japan", "Japan is known for its technology, culture, and Mount Fuji."),
        ("Tell me about Canada", "Canada is famous for maple syrup, beautiful landscapes, and friendly people."),
        ("Wow", "Amazing, right?"),
        ("Cool", "Glad you liked it!"),
        ("Nice", "Thanks!"),
        ("Okay", "Got it!"),
        ("Alright", "Let’s keep going!"),
        ("Really?", "Yup! Absolutely."),
        ("Are you sure?", "I'm pretty confident!"),
        ("That's funny", "I try my best!"),
        ("You're smart", "Thank you! You’re not so bad yourself."),
        ("You're funny", "You're making me blush... if I could."),
        ("I like you", "I like chatting with you too!")
    };
}
*/


// *** Inicialização Final ***
// ... (Mapeamento de /chat e HandleWebSocketAsync como antes) ...
// ... (Função GetTrainingData como antes) ...

Console.WriteLine("Setup complete. Starting the web server...");
await app.RunAsync(); // Mantém o servidor rodando