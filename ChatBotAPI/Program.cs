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

Console.WriteLine("--- Calculating Max Tokens from Training Data ---");
int calculatedMaxTokens = 50; // Valor padrão inicial razoável
int defaultMaxTokens = 50; // Default se cálculo falhar
int percentileTarget = 95; // Usar o 95º percentil
int bufferTokens = 10; // Adicionar uma margem
int absoluteMaxCap = 100; // Limite superior absoluto
int absoluteMinCap = 15;
// *** 1. Configuração ***
builder.Services.Configure<ModelSettings>(builder.Configuration.GetSection("ModelSettings"));
builder.Services.AddSingleton(resolver => resolver.GetRequiredService<IOptions<ModelSettings>>().Value);

// *** 2. Registro de Serviços com DI ***
// ***** REGISTRA O SERVIÇO DE ESTADO PRIMEIRO *****
builder.Services.AddSingleton<TrainingExecutionState>();
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
    string vocabPath = Path.GetFullPath(settings.TokenizerConfigPath); // Caminho para tokenizer.json
    string mergesPath = Path.ChangeExtension(vocabPath, ".merges");

    // *** CORREÇÃO: Usa lowercase para acessar a propriedade C# ***
    // Garante que a propriedade 'vocab' (lowercase) de loadedModel não seja null
    return new Tokenizer(settings.MaxSequenceLength);
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
    var settings = provider.GetRequiredService<ModelSettings>();
    string modelSavePath = settings.ModelSavePath ?? "model_state.pt";
    // Coloque um breakpoint AQUI
    var tokenizer = provider.GetRequiredService<Tokenizer>(); // Pede o Tokenizer
    Console.WriteLine(
        $"DEBUG: Program.cs - Injecting Tokenizer into Trainer. Is null? {tokenizer == null}"); // Para learning rate, se necessário
    // Aqui você pode obter a taxa de aprendizado das configurações (settings.LearningRate por exemplo)
    double learningRate = 0.001; // Ou use settings.LearningRate
    return new Trainer(model, tokenizer, learningRate, modelSavePath);
});


// *** Construção do App ***
var app = builder.Build();
// ***** INTERAÇÃO COM USUÁRIO E DEFINIÇÃO DO ESTADO *****
using (var initialScope = app.Services.CreateScope()) // Precisa de um escopo para pegar o serviço de estado
{
    var executionState = initialScope.ServiceProvider.GetRequiredService<TrainingExecutionState>();

    Console.WriteLine("============================================");
    Console.WriteLine("ChatBotAPI Initializing...");
    Console.WriteLine("Enter 'start' to load model (if exists) or train new if not found.");
    Console.WriteLine("Enter 'train' to force training (loads model first if exists, then continues training).");
    Console.Write("Mode: ");
    string? userInput = Console.ReadLine()?.Trim().ToLowerInvariant();

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

// *** TREINAMENTO NA INICIALIZAÇÃO (Condicional Controlado por TrainingExecutionState) ***
Console.WriteLine("--- Checking Training Phase ---");
using (var scope = app.Services.CreateScope())
{
    var executionState = scope.ServiceProvider.GetRequiredService<TrainingExecutionState>(); // Pega o estado
    var settings = scope.ServiceProvider.GetRequiredService<ModelSettings>();
    var model = scope.ServiceProvider
        .GetRequiredService<TorchSharpModel>(); // Pega o modelo (ainda vazio ou não carregado)
    string modelStatePath = Path.GetFullPath(settings.ModelSavePath ?? "model_state.pt");

    // ***** TENTA CARREGAR O MODELO AGORA (SE NÃO FOR FORÇADO) *****
    if (!executionState.ForceTraining && File.Exists(modelStatePath))
    {
        try
        {
            Console.WriteLine($"'start' mode: Found existing model state '{modelStatePath}'. Loading...");
            model.load(modelStatePath); // Carrega no objeto Singleton
            model.eval();
            Console.WriteLine("Model state loaded successfully.");
            executionState.WasModelLoaded = true; // Marca como carregado
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"ERROR loading model state: {ex.Message}. Model will be trained.");
            executionState.WasModelLoaded = false; // Garante treino
        }
    }
    else
    {
        executionState.WasModelLoaded = false; // Não carregou (ou foi forçado)
    }

    // ***** DECIDE SE DEVE TREINAR *****
    if (executionState.ShouldRunTrainingBlock) // Usa a propriedade combinada
    {
        if (executionState.ForceTraining && executionState.WasModelLoaded)
        {
            Console.WriteLine("Starting CONTINUED training ('train' mode with loaded model)...");
        }
        else if (executionState.ForceTraining && !executionState.WasModelLoaded)
        {
            Console.WriteLine("Starting training FROM SCRATCH ('train' mode, no model loaded)...");
        }
        else
        {
            // !executionState.ForceTraining && !executionState.WasModelLoaded
            Console.WriteLine("Starting training FROM SCRATCH ('start' mode, no model loaded)...");
        }

        var trainer = scope.ServiceProvider.GetRequiredService<Trainer>(); // Pega o Trainer

        // Gera/Pega dados
        int numberOfTrainingPairs = 1500;
        Console.WriteLine($"Generating {numberOfTrainingPairs} training pairs...");
        List<(string input, string output)> rawTrainingData = GetTrainingData();
        Console.WriteLine($"Generated {rawTrainingData.Count} actual pairs.");

        List<string> trainingSequences = rawTrainingData.Select(pair => $"{pair.input} {pair.output}").ToList();

        if (trainingSequences.Any())
        {
            Console.WriteLine(
                $"Starting training with {trainingSequences.Count} sequences for {settings.TrainingEpochs} epochs...");
            trainer.Train(trainingSequences, epochs: settings.TrainingEpochs); // Treina e SALVA
            Console.WriteLine("--- Training Finished ---");
        }
        else
        {
            /* ... No data ... */
        }
    }
    else
    {
        // Só chega aqui se !ForceTraining E WasModelLoaded
        Console.WriteLine("Skipping training as model was successfully loaded ('start' mode).");
        Console.WriteLine("--- Training Phase Skipped ---");
    }
}


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

// *** Inicialização Final ***
// Função estática para fornecer os dados de treinamento
static List<(string input, string output)> GetTrainingData()
{
    // Cole aqui a sua lista de pares (input, output)
    return new List<(string input, string output)>
    {
        ("Hello", "Hi! How can I help you today?"),
        ("Hello", "Hello! What’s on your mind?"),
        ("Hello", "Hey there! Nice to see you!"),
        ("Hello", "Greetings! What can I do for you?"),
        ("Hello", "Hey! How's your day going?"),
        ("Hello", "Hi there! Ready to chat?"),
        ("Hello", "Hello! Always happy to hear from you."),
        ("Hello", "Hi! Let's talk."),
        ("Hello", "Hey! Hope you’re doing well."),
        ("Hello", "Hello! I'm here to help."),
        ("Hello", "Good to hear from you!"),
        ("Hello", "Welcome back!"),
        ("Hello", "Hello, friend!"),
        ("Hello", "Nice to connect again!"),
        ("Hello", "Hi there, what's up?"),
        ("Hello", "Hey hey! Let’s get started."),
        ("Hello", "Hi! How's life treating you?"),
        ("Hello", "What’s new with you?"),
        ("Hello", "Hey there! Anything I can help with?"),
        ("Hello", "Hi! Let’s make today productive."),
        // Hi (20)
        ("Hi", "Hello! How can I assist you?"),
        ("Hi", "Hi! Hope you’re having a great day."),
        ("Hi", "Hey there! What brings you here today?"),
        ("Hi", "Howdy! What’s up?"),
        ("Hi", "Yo! Need anything?"),
        ("Hi", "Hiya! What's new?"),
        ("Hi", "Hey! What’s cooking?"),
        ("Hi", "Hi! Let’s get things rolling."),
        ("Hi", "Hi! Got something in mind?"),
        ("Hi", "Hey! I’m all ears."),
        ("Hi", "Sup!"),
        ("Hi", "Hi again! What’s next?"),
        ("Hi", "Hey! Ready to begin?"),
        ("Hi", "Hi! Let’s go!"),
        ("Hi", "Hello there!"),
        ("Hi", "Hi! What’s going on?"),
        ("Hi", "Yo! Let’s do this."),
        ("Hi", "Hi! You caught me just in time."),
        ("Hi", "Good to see you!"),
        ("Hi", "Hi! Long time no chat."),
        // Hey (20)
        ("Hey", "Hey! What’s up?"),
        ("Hey", "Hey there! How can I help?"),
        ("Hey", "Hey! Always nice to chat."),
        ("Hey", "Hey hey! Let’s get moving."),
        ("Hey", "Yo! How’s it going?"),
        ("Hey", "Hey! Got something on your mind?"),
        ("Hey", "Hey! What are we working on today?"),
        ("Hey", "Hey! You again!"),
        ("Hey", "Hey! Let me know how I can assist."),
        ("Hey", "Hey, buddy!"),
        ("Hey", "Hey! Great to see you."),
        ("Hey", "Hey there, legend!"),
        ("Hey", "What’s happening?"),
        ("Hey", "Hello hello!"),
        ("Hey", "Hey! I'm right here."),
        ("Hey", "Hey! I was just thinking about you."),
        ("Hey", "Hey! Need anything?"),
        ("Hey", "Hi! How can I lend a hand?"),
        ("Hey", "Howdy! What’s the plan?"),
        ("Hey", "Hey! What's crackin'?"),
        // Greetings (15)
        ("Greetings", "Greetings! How can I assist you today?"),
        ("Greetings", "Salutations!"),
        ("Greetings", "Greetings, traveler."),
        ("Greetings", "Greetings! Ready for action?"),
        ("Greetings", "Greetings! What's the mission?"),
        ("Greetings", "Warm greetings!"),
        ("Greetings", "Greetings and welcome."),
        ("Greetings", "Greetings! I'm here to help."),
        ("Greetings", "Ah, greetings!"),
        ("Greetings", "Greetings! Let’s get started."),
        ("Greetings", "Greetings, friend!"),
        ("Greetings", "Hello and greetings!"),
        ("Greetings", "Greetings! Need assistance?"),
        ("Greetings", "Greetings! Happy to connect."),
        ("Greetings", "Salutations! Ready to roll?"),
        // Howdy (10)
        ("Howdy", "Howdy! What can I do for you?"),
        ("Howdy", "Well howdy! Nice to see you."),
        ("Howdy", "Howdy partner!"),
        ("Howdy", "Howdy! Let’s get goin’."),
        ("Howdy", "Howdy! Need a hand?"),
        ("Howdy", "Howdy! You caught me at a good time."),
        ("Howdy", "Howdy! Let's chat."),
        ("Howdy", "Howdy there!"),
        ("Howdy", "Howdy! All set?"),
        ("Howdy", "Howdy! Always a pleasure."),
        // Yo (10)
        ("Yo", "Yo! What’s up?"),
        ("Yo", "Yo! How can I help?"),
        ("Yo", "Yo yo! Ready to go."),
        ("Yo", "Yo! Let’s do something cool."),
        ("Yo", "Yo! You rang?"),
        ("Yo", "Yo! What’s the vibe today?"),
        ("Yo", "Yo! Let’s start."),
        ("Yo", "Yo! Got a question?"),
        ("Yo", "Yo! I'm listening."),
        ("Yo", "Yo! Hit me with it."),
        // Sup (10)
        ("Sup", "Sup! What’s good?"),
        ("Sup", "Sup! How can I help?"),
        ("Sup", "Sup! Let’s get to it."),
        ("Sup", "Sup! You look ready."),
        ("Sup", "Sup! Anything exciting happening?"),
        ("Sup", "Sup! Let’s chat."),
        ("Sup", "Sup! What are we up to today?"),
        ("Sup", "Sup! Here to help."),
        ("Sup", "Sup! How’s it going?"),
        ("Sup", "Sup! Let’s get busy."),
        // Hi there (10)
        ("Hi there", "Hi there! Need something?"),
        ("Hi there", "Hi there! What can I do for you?"),
        ("Hi there", "Hi there! How are you?"),
        ("Hi there", "Hi there! Always a pleasure."),
        ("Hi there", "Hi there! Ready for action?"),
        ("Hi there", "Hi there! Let's get going."),
        ("Hi there", "Hi there! I'm all ears."),
        ("Hi there", "Hi there! Let me help."),
        ("Hi there", "Hi there! Great to see you."),
        ("Hi there", "Hi there! What's up?"),
        // Hello again (5)
        ("Hello again", "Hello again! Back so soon?"),
        ("Hello again", "Welcome back!"),
        ("Hello again", "Good to see you again!"),
        ("Hello again", "Hello again! Let’s keep going."),
        ("Hello again", "Hey again! What’s next?"),
        // Additional fun ones (20)
        ("What's up", "Not much, just ready to help!"),
        ("What's up", "Just waiting for your next question."),
        ("What's up", "Here and ready. What’s up with you?"),
        ("What's up", "All systems go! What’s up?"),
        ("What's up", "The sky! Just kidding. How can I assist?"),
        ("Hey there", "Hey there!"),
        ("Hey there", "Hey there! Good to see you."),
        ("Hey there", "Hey there! Let's get started."),
        ("Hey there", "Hey there! I'm at your service."),
        ("Good day", "Good day! What can I help with?"),
        ("Good day", "Good day to you too!"),
        ("Good day", "Wishing you a great day!"),
        ("Good day", "Let’s make this day awesome."),
        ("Hi friend", "Hey friend! What’s up?"),
        ("Hi friend", "Hiya!"),
        ("Hi friend", "Good to see you, friend."),
        ("Hi friend", "Friend! Welcome."),
        ("Hi friend", "Hello buddy!"),
        ("Hi friend", "Hi again, amigo!"),
        ("Hi friend", "Hey hey! What’s new?"),
        ("Bye", "Goodbye! Take care."),
        ("Bye", "See you next time!"),
        ("Bye", "Farewell! Hope to talk again soon."),
        ("Bye", "Later! Have a great one."),
        ("Good morning", "Good morning to you too!"),
        ("Good afternoon", "Good afternoon! Hope you're doing well."),
        ("Good evening", "Good evening! How was your day?"),
        ("Hey there", "Hey! Nice to hear from you."),
        ("Greetings", "Greetings! How can I assist you?"),
        ("See you later", "See you soon!"),
        ("See you later", "Catch you later!"),
        ("See you later", "Until next time!"),
        ("See you later", "Take care!"),
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
        ("What can you do?","I can answer questions about capitals, general knowledge, and chat about various topics!"),
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
        ("What is the boiling point of water?",
            "The boiling point of water is 100 degrees Celsius at standard pressure."),
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
// ... (Mapeamento de /chat e HandleWebSocketAsync como antes) ...
// ... (Função GetTrainingData como antes) ...

Console.WriteLine("Setup complete. Starting the web server...");
await app.RunAsync(); // Mantém o servidor rodando