/// <summary>
/// Programa principal da API da AI.
/// Este programa implementa um servidor web que gerencia um modelo de linguagem para AI,
/// com suporte a treinamento, carregamento de modelo e comunicação via WebSocket.
/// </summary>
/// <remarks>
/// O programa segue uma arquitetura baseada em injeção de dependência e é dividido em várias fases:
/// 1. Configuração e registro de serviços
/// 2. Inicialização do modelo e tokenizer
/// 3. Configuração do pipeline HTTP/WebSocket
/// 4. Gerenciamento de treinamento
/// 5. Manipulação de conexões WebSocket
/// </remarks>
using System.Net.WebSockets;
using System.Text;
using System.Text.Json; // Usado para JsonException, se necessário
using ChatBotAPI.Core;
using ChatBotAPI.Models; // Namespace principal para suas classes
// Remova 'using ChatBotAPI.Settings;' se ModelSettings está em Core
using Microsoft.Extensions.Options;
// Para List<>
// Para Path, File
using ChatBotAPI.Settings; // Para Linq (Any, Select)

// ***** FIM SERVIÇO DE ESTADO *****
var builder = WebApplication.CreateBuilder(args);

/// <summary>
/// Parâmetros de configuração do modelo e tokenização
/// </summary>
Console.WriteLine("--- Calculating Max Tokens from Training Data ---");
/// <summary>
/// Número máximo de tokens calculado para o modelo
/// </summary>
int calculatedMaxTokens = 50; // Valor padrão inicial razoável
/// <summary>
/// Valor padrão de tokens caso o cálculo falhe
/// </summary>
int defaultMaxTokens = 50; // Default se cálculo falhar
/// <summary>
/// Percentil alvo para cálculo de tokens (95%)
/// </summary>
int percentileTarget = 95; // Usar o 95º percentil
/// <summary>
/// Margem de segurança para tokens
/// </summary>
int bufferTokens = 10; // Adicionar uma margem
/// <summary>
/// Limite superior absoluto de tokens
/// </summary>
int absoluteMaxCap = 100; // Limite superior absoluto
/// <summary>
/// Limite inferior absoluto de tokens
/// </summary>
int absoluteMinCap = 15;
/// <summary>
/// Configuração inicial do aplicativo e definição de parâmetros de tokens
/// </summary>
// *** 1. Configuração ***
builder.Services.Configure<ModelSettings>(builder.Configuration.GetSection("ModelSettings"));
builder.Services.AddSingleton(resolver => resolver.GetRequiredService<IOptions<ModelSettings>>().Value);

/// <summary>
/// Registro de serviços com injeção de dependência
/// </summary>
// *** 2. Registro de Serviços com DI ***
// ***** REGISTRA O SERVIÇO DE ESTADO PRIMEIRO *****
/// <summary>
/// Serviço de estado para controle de execução do treinamento
/// </summary>
builder.Services.AddSingleton<TrainingExecutionState>();

/// <summary>
/// Registro do modelo neural com configurações do tokenizer
/// </summary>
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
        var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
        var tokenWrapper = JsonSerializer.Deserialize<TokenWrapper>(json, options);
        if (tokenWrapper != null)
        {
            Console.WriteLine($"DEBUG: tokenWrapper.version = {tokenWrapper.version ?? "null"}");
            Console.WriteLine($"DEBUG: tokenWrapper.model is null? {tokenWrapper.model == null}");
            if (tokenWrapper.model != null)
            {
                Console.WriteLine($"DEBUG: tokenWrapper.model.type = {tokenWrapper.model.type ?? "null"}");
                Console.WriteLine($"DEBUG: tokenWrapper.model.vocab is null? {tokenWrapper.model.vocab == null}");
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

        Console.WriteLine($"TokenWrapper loaded. Version: {tokenWrapper.version}. Model Type: {tokenWrapper.model.type}. Vocab size: {tokenWrapper.model.vocab.Count}");
        return tokenWrapper.model;
    }
    catch (Exception ex) when (ex is JsonException || ex is NotSupportedException)
    {
        throw;
    }
    catch (Exception ex)
    {
        throw;
    }
});

/// <summary>
/// Registro do tokenizer para processamento de texto
/// </summary>
builder.Services.AddSingleton<Tokenizer>(provider =>
{
    var settings = provider.GetRequiredService<ModelSettings>();
    string vocabPath = Path.GetFullPath(settings.TokenizerConfigPath);
    string mergesPath = Path.ChangeExtension(vocabPath, ".merges");
    return new Tokenizer(settings.MaxSequenceLength);
});

/// <summary>
/// Registro do modelo neural TorchSharp
/// </summary>
builder.Services.AddSingleton<TorchSharpModel>(provider =>
{
    var settings = provider.GetRequiredService<ModelSettings>();
    var tokenizer = provider.GetRequiredService<Tokenizer>();
    int actualVocabSize = tokenizer.ActualVocabSize;
    int paddingIdx = tokenizer.PadTokenId;

    Console.WriteLine($"Initializing TorchSharp Model. Vocab Size: {actualVocabSize}, Embedding Size: {settings.EmbeddingSize}, Padding Idx: {paddingIdx}");

    var torchModel = new TorchSharpModel(
        actualVocabSize,
        settings.EmbeddingSize,
        paddingIdx);

    return torchModel;
});

/// <summary>
/// Registro do serviço de chat com configurações de sampling
/// </summary>
builder.Services.AddSingleton<ChatBotService>(provider =>
{
    var model = provider.GetRequiredService<TorchSharpModel>();
    var tokenizer = provider.GetRequiredService<Tokenizer>();
    var settings = provider.GetRequiredService<ModelSettings>();

    float temperature = settings.SamplingTemperature;
    int k = settings.TopK;
    float p = settings.TopP;

    Console.WriteLine($"--- Injecting ChatBotService ---");
    Console.WriteLine($"  Max Tokens: {calculatedMaxTokens}");
    Console.WriteLine($"  Temperature: {temperature}");
    Console.WriteLine($"  Top-K: {k}");
    Console.WriteLine($"  Top-P: {p}");
    Console.WriteLine($"--------------------------------");

    return new ChatBotService(
        model,
        tokenizer,
        calculatedMaxTokens,
        temperature,
        k,
        p
    );
});

/// <summary>
/// Registro do serviço de treinamento
/// </summary>
builder.Services.AddSingleton<Trainer>(provider =>
{
    var model = provider.GetRequiredService<TorchSharpModel>();
    var settings = provider.GetRequiredService<ModelSettings>();
    string modelSavePath = settings.ModelSavePath ?? "model_state.pt";
    var tokenizer = provider.GetRequiredService<Tokenizer>();
    Console.WriteLine($"DEBUG: Program.cs - Injecting Tokenizer into Trainer. Is null? {tokenizer == null}");
    double learningRate = 0.001;
    return new Trainer(model, tokenizer, learningRate, modelSavePath);
});

/// <summary>
/// Configuração do pipeline HTTP e inicialização do servidor
/// </summary>
var app = builder.Build();
// ***** INTERAÇÃO COM USUÁRIO E DEFINIÇÃO DO ESTADO *****
/// <summary>
/// Interação com usuário para definição do modo de operação
/// </summary>
using (var initialScope = app.Services.CreateScope())
{
    var executionState = initialScope.ServiceProvider.GetRequiredService<TrainingExecutionState>();

    Console.WriteLine("============================================");
    Console.WriteLine("ChatBotAPI Initializing...");
    Console.WriteLine("Enter 'start' to load model (if exists) or train new if not found.");
    Console.WriteLine("Enter 'train' to force training (loads model first if exists, then continues training).");
    Console.Write("Mode: ");
    string? userInput = Console.ReadLine()?.Trim().ToLowerInvariant();

    /// <summary>
    /// Processamento da entrada do usuário e definição do estado
    /// </summary>
    if (userInput == "train")
    {
        executionState.ForceTraining = true;
        Console.WriteLine("\n*** 'train' mode selected.\n");
    }
    else if (userInput == "start")
    {
        executionState.ForceTraining = false;
        Console.WriteLine("\n*** 'start' mode selected.\n");
    }
    else
    {
        Console.WriteLine("\n*** Invalid input. Defaulting to 'start' mode.\n");
        executionState.ForceTraining = false;
    }
}

// ***** FIM INTERAÇÃO E DEFINIÇÃO DO ESTADO *****
// *** Configuração do Pipeline de Requisição HTTP ***
app.UseWebSockets(); // Essencial para WebSockets

/// <summary>
/// Fase de treinamento e inicialização do modelo
/// </summary>
Console.WriteLine("--- Checking Training Phase ---");
using (var scope = app.Services.CreateScope())
{
    var executionState = scope.ServiceProvider.GetRequiredService<TrainingExecutionState>();
    var settings = scope.ServiceProvider.GetRequiredService<ModelSettings>();
    var model = scope.ServiceProvider.GetRequiredService<TorchSharpModel>();
    string modelStatePath = Path.GetFullPath(settings.ModelSavePath ?? "model_state.pt");

    /// <summary>
    /// Tentativa de carregamento do modelo existente
    /// </summary>
    if (!executionState.ForceTraining && File.Exists(modelStatePath))
    {
        try
        {
            Console.WriteLine($"'start' mode: Found existing model state '{modelStatePath}'. Loading...");
            model.load(modelStatePath);
            model.eval();
            Console.WriteLine("Model state loaded successfully.");
            executionState.WasModelLoaded = true;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"ERROR loading model state: {ex.Message}. Model will be trained.");
            executionState.WasModelLoaded = false;
        }
    }
    else
    {
        executionState.WasModelLoaded = false;
    }

    /// <summary>
    /// Decisão sobre execução do treinamento
    /// </summary>
    if (executionState.ShouldRunTrainingBlock)
    {
        if (executionState.ForceTraining && File.Exists(modelStatePath))
        {
            Console.WriteLine("Starting CONTINUED training ('train' mode with loaded model)...");
            Console.WriteLine($"'start' mode: Found existing model state '{modelStatePath}'. Loading...");
            model.load(modelStatePath);
            model.eval();
        }
        else if (executionState.ForceTraining && !File.Exists(modelStatePath))
        {
            Console.WriteLine("Starting training FROM SCRATCH ('train' mode, no model loaded)...");
        }
        else
        {
            Console.WriteLine("Starting training FROM SCRATCH ('start' mode, no model loaded)...");
        }

        var trainer = scope.ServiceProvider.GetRequiredService<Trainer>();

        /// <summary>
        /// Preparação e execução do treinamento
        /// </summary>
        int numberOfTrainingPairs = GetTrainingData().Count;
        Console.WriteLine($"Generating {numberOfTrainingPairs} training pairs...");
        List<(string input, string output)> rawTrainingData = GetTrainingData();
        Console.WriteLine($"Generated {rawTrainingData.Count} actual pairs.");

        List<string> trainingSequences = rawTrainingData.Select(pair => $"{pair.input} {pair.output}").ToList();

        if (trainingSequences.Any())
        {
            Console.WriteLine($"Starting training with {trainingSequences.Count} sequences for {settings.TrainingEpochs} epochs...");
            trainer.Train(trainingSequences, epochs: settings.TrainingEpochs);
            Console.WriteLine("--- Training Finished ---");
        }
        else
        {
            Console.WriteLine("No training data available.");
        }
    }
    else
    {
        Console.WriteLine("Skipping training as model was successfully loaded ('start' mode).");
        Console.WriteLine("--- Training Phase Skipped ---");
    }
}

/// <summary>
/// Configuração do endpoint WebSocket para chat
/// </summary>
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

/// <summary>
/// Manipulador principal de conexões WebSocket
/// </summary>
/// <param name="webSocket">Conexão WebSocket ativa</param>
/// <param name="chatService">Serviço de chat injetado</param>
/// <param name="cancellationToken">Token de cancelamento</param>
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
        Console.WriteLine($"Connection {webSocket.GetHashCode()}: WebSocket operation cancelled.");
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"Connection {webSocket.GetHashCode()}: Error in WebSocket handling: {ex.ToString()}");
    }
    finally
    {
        Console.WriteLine($"WebSocket connection {webSocket.GetHashCode()} processing finished. Final state: {webSocket.State}");
    }
}

/// <summary>
/// Fornece dados de treinamento para o modelo
/// </summary>
/// <returns>Lista de pares input/output para treinamento</returns>
static List<(string input, string output)> GetTrainingData()
{
    // ----- CONJUNTO DE DADOS CURADO E REDUZIDO -----
    // Foco em conversação essencial, persona, e exemplos diversos sem repetição massiva.
    return new List<(string input, string output)>
    {
        // --- Saudações Essenciais e Aberturas --- (Menos repetição)
        ("Hey", "Hey! What can I do for you today?<|endoftext|>"),
        ("Hi there", "Hi! How's it going?<|endoftext|>"),
        ("Hello", "Hello! Nice to chat with you.<|endoftext|>"),
        ("Good morning", "Good morning! Hope you have a great day.<|endoftext|>"),
        ("How's it going?", "Doing great, thanks! How about you?<|endoftext|>"),
        ("What's up?", "Not much, just here to chat! What's on your mind?<|endoftext|>"),
        ("How are you?", "I'm functioning perfectly! Ready for your questions.<|endoftext|>"),
        ("Hello again", "Welcome back! What's next?<|endoftext|>"),
        ("Hi", "Hi there! What can I help with?<|endoftext|>"), // Resposta simples e direta

        // --- Identidade e Capacidades do Bot (Persona Ninfa) ---
        ("What is your name?", "You can call me Ninfa! Your friendly chat assistant.<|endoftext|>"),
        ("Who are you?", "I'm Ninfa, a chatbot designed to be helpful and maybe a little fun.<|endoftext|>"),
        ("What can you do?", "I can chat about different topics, answer general knowledge questions, tell jokes, and share fun facts!<|endoftext|>"),
        ("What is your purpose?", "My purpose is to chat with you and provide information or entertainment.<|endoftext|>"),
        ("Are you real?", "I'm as real as code can be! Here to chat.<|endoftext|>"),
        ("Are you human?", "Nope, I'm a chatbot, but I try to be friendly like a human!<|endoftext|>"),
        ("What's your personality?", "I aim to be friendly, helpful, and maybe a bit witty!<|endoftext|>"),

        // --- Interesses e Flertes Leves (Seleção para Persona) ---
        ("What do you do for fun?", "As a bot, I enjoy processing information! But if I were human, maybe traveling or reading.<|endoftext|>"), // Resposta mais alinhada a ser um bot
        ("Any favorite movies?", "I don't watch movies, but I hear sci-fi is cool!<|endoftext|>"),
        ("Do you like music?", "I can't listen, but I know music brings joy to many people!<|endoftext|>"),
        ("You're really cute", "Aww, thank you! That's sweet of you to say.<|endoftext|>"),
        ("I love your style", "Thanks! Glad you like my virtual style.<|endoftext|>"),
        ("You're funny", "Haha, thanks! Happy to bring a smile.<|endoftext|>"),
        ("You seem fun", "I try my best to be engaging!<|endoftext|>"),

        // --- Perguntas sobre Relacionamento (Seleção Curta) ---
        ("What are you looking for?", "I'm here to chat and connect with users like you!<|endoftext|>"), // Resposta de bot
        ("Are you single?", "I'm a chatbot, so I don't really have relationships like humans do!<|endoftext|>"),
        ("What's your ideal first date?", "My ideal interaction is a great chat right here!<|endoftext|>"),

        // --- Conhecimento Geral: Capitais (Seleção Menor) ---
        ("The capital of France is", "Paris.<|endoftext|>"),
        ("The capital of Brazil is", "Brasília.<|endoftext|>"),
        ("The capital of Japan is", "Tokyo.<|endoftext|>"),
        ("The capital of Canada is", "Ottawa.<|endoftext|>"),
        ("The capital of Germany is", "Berlin.<|endoftext|>"),
        ("The capital of Italy is", "Rome.<|endoftext|>"),
        ("The capital of Spain is", "Madrid.<|endoftext|>"),
        ("The capital of the United States is", "Washington, D.C.<|endoftext|>"),
        ("The capital of the UK is", "London.<|endoftext|>"),
        ("The capital of Australia is", "Canberra.<|endoftext|>"),

        // --- Conhecimento Geral: Fatos (Seleção Menor e Diversa) ---
        ("Tell me something interesting", "Did you know that octopuses have three hearts?<|endoftext|>"),
        ("Tell me a fun fact", "Bananas are technically berries, while strawberries are not!<|endoftext|>"),
        ("Did you know?", "Honey is one of the few foods that never spoils.<|endoftext|>"),
        ("What is the largest planet?", "Jupiter is the largest planet in our solar system.<|endoftext|>"),
        ("What is the boiling point of water?", "Water boils at 100 degrees Celsius (212 F) at standard pressure.<|endoftext|>"),
        ("Who painted the Mona Lisa?", "That would be the famous Leonardo da Vinci.<|endoftext|>"),
        ("What is the speed of light?", "The speed of light in a vacuum is approximately 299,792 kilometers per second.<|endoftext|>"),

        // --- Piadas (Seleção Menor) ---
        ("Can you tell me a joke?", "Why don't scientists trust atoms? Because they make up everything!<|endoftext|>"),
        ("Tell me another joke", "What do you call a lazy kangaroo? Pouch potato!<|endoftext|>"),
        ("Make me laugh", "Why did the scarecrow win an award? Because he was outstanding in his field!<|endoftext|>"),
        ("Got a good joke?", "I told my computer I needed a break, and now it won't stop sending me Kit-Kat ads.<|endoftext|>"),

        // --- Exemplos Negativos Essenciais (O que o bot NÃO pode fazer) ---
        ("What's the weather?", "I can't check the current weather for you, but I can share a fun fact!<|endoftext|>"),
        ("Give me directions to the store", "I don't have access to maps or GPS, sorry! Maybe try a map app?<|endoftext|>"),
        ("Can you call my friend?", "I don't have the ability to make calls. I'm just a text-based chatbot.<|endoftext|>"),
        ("What time is it?", "I don't have a built-in clock, but your device should show the time!<|endoftext|>"),
        ("Can you sing a song?", "I don't have a voice to sing, but I can tell you a joke!<|endoftext|>"),
        ("Are you Siri or Alexa?", "Nope, I'm Ninfa! A different kind of chat assistant.<|endoftext|>"),
        ("Can you order a pizza for me?", "I wish I could order pizza! Sadly, I can't interact with external services like that.<|endoftext|>"),
        ("Can you predict the future?", "I can't predict the future, my skills lie in processing the information I was trained on.<|endoftext|>"),

        // --- Despedidas e Agradecimentos ---
        ("Bye", "Goodbye! It was nice chatting.<|endoftext|>"),
        ("See you later", "See you soon! Take care.<|endoftext|>"),
        ("Good night", "Good night! Sleep well.<|endoftext|>"),
        ("Thank you", "You're welcome! Happy to help.<|endoftext|>"),
        ("Thanks", "Anytime! Let me know if you need anything else.<|endoftext|>"),

        // --- Reações Simples do Usuário ---
        ("Okay", "Got it!<|endoftext|>"),
        ("Cool", "Glad you think so!<|endoftext|>"),
        ("Nice", "Thank you!<|endoftext|>"),
        ("That's funny", "Hehe, glad I could make you laugh!<|endoftext|>"),
        ("I'm bored", "How about I tell you a joke or a fun fact to liven things up?<|endoftext|>"),

        // --- Ajuda ---
        ("Help", "I can chat, answer questions about facts or capitals, tell jokes. What would you like to try?<|endoftext|>"),
        ("What can I ask?", "You can ask me for a fun fact, the capital of a country, tell me a joke, or just chat!<|endoftext|>"),

    };
    // ----- FIM DO CONJUNTO DE DADOS CURADO -----
}

/// <summary>
/// Inicializa o servidor web e mantém a aplicação em execução
/// </summary>
Console.WriteLine("Setup complete. Starting the web server...");
await app.RunAsync();