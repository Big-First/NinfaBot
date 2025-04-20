using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using ChatBotAPI.Core;
using ChatBotAPI.enums;
// using ChatBotAPI.Models; // Não é mais necessário para TokenWrapper
using Microsoft.Extensions.Options;
using ChatBotAPI.Settings;
using TorchSharp;
using static TorchSharp.torch; // Adicionado para torch.load

var builder = WebApplication.CreateBuilder(args);

// *** 1. Configuração ***
builder.Services.Configure<ModelSettings>(builder.Configuration.GetSection("ModelSettings"));
builder.Services.AddSingleton(resolver => resolver.GetRequiredService<IOptions<ModelSettings>>().Value);

// *** 2. Registro de Serviços com DI ***
builder.Services.AddSingleton<TrainingExecutionState>();

// --- Tokenizer (usando SharpToken) ---
builder.Services.AddSingleton<Tokenizer>(provider =>
{
    var settings = provider.GetRequiredService<ModelSettings>();
    // Construtor SharpToken só precisa de MaxSequenceLength
    return new Tokenizer(settings.MaxSequenceLength);
});

// --- NeuralModel (TransformerModel) ---
builder.Services.AddSingleton<TransformerModel>(provider =>
{
    var settings = provider.GetRequiredService<ModelSettings>();
    var tokenizer = provider.GetRequiredService<Tokenizer>();

    int vocabSize = tokenizer.VocabSize;
    int paddingIdx = tokenizer.PadTokenId;

    if (settings.DModel % settings.Nhead != 0)
    {
        throw new ArgumentException("DModel must be divisible by Nhead");
    }

    Console.WriteLine($"Initializing Transformer Model:");
    Console.WriteLine($"  Vocab Size : {vocabSize}");
    Console.WriteLine($"  DModel     : {settings.DModel}");
    Console.WriteLine($"  Nhead      : {settings.Nhead}");
    Console.WriteLine($"  Num Layers : {settings.NumDecoderLayers}");
    Console.WriteLine($"  Dim FF     : {settings.DimFeedforward}");
    Console.WriteLine($"  Dropout    : {settings.DropoutRate}");
    Console.WriteLine($"  Padding Idx: {paddingIdx}");

    var transformerModel = new TransformerModel(
        vocabSize: vocabSize,
        dModel: settings.DModel,
        nhead: settings.Nhead,
        numDecoderLayers: settings.NumDecoderLayers,
        dimFeedforward: settings.DimFeedforward,
        dropoutRate: settings.DropoutRate,
        paddingIdx: paddingIdx
    );
    // Definir o device inicial do modelo (CPU por padrão, mas pode ser movido depois)
    // var initialDevice = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
    // transformerModel.to(initialDevice); // Mover parâmetros
    // transformerModel.SetDevice(initialDevice); // Informar o modelo sobre seu device
    return transformerModel;
});

// --- Trainer ---
builder.Services.AddSingleton<Trainer>(provider =>
{
    var model = provider.GetRequiredService<TransformerModel>(); // Pede TransformerModel
    var settings = provider.GetRequiredService<ModelSettings>();
    var tokenizer = provider.GetRequiredService<Tokenizer>();
    string modelSavePath = settings.ModelSavePath ?? "model_transformer_state.pt"; // Novo nome padrão
    double learningRate = 0.001; // TODO: Ler das settings? settings.LearningRate

    Console.WriteLine($"DEBUG: Program.cs - Injecting Tokenizer into Trainer. Is null? {tokenizer == null}");

    // *** PASSA TransformerModel ***
    return new Trainer(model, tokenizer, learningRate, modelSavePath);
});


// ***********************************************************************
// *** CÁLCULO DE MAX TOKENS - DEPOIS DOS REGISTROS ESSENCIAIS ***
// ***********************************************************************
Console.WriteLine("--- Calculating Max Tokens from Training Data ---");
int calculatedMaxTokens = 50; // Valor padrão inicial ou o valor do teste anterior
int defaultMaxTokens = 50;
int percentileTarget = 95;
int bufferTokens = 10;
int absoluteMaxCap = 100; // Ajuste conforme necessário
int absoluteMinCap = 15; // Ajuste conforme necessário

// Cria um provedor de serviços temporário APENAS para obter o Tokenizer
var tempServiceProvider = builder.Services.BuildServiceProvider();
using (var tempScope = tempServiceProvider.CreateScope())
{
    var tokenizer = tempScope.ServiceProvider.GetRequiredService<Tokenizer>();
    List<(string input, string output)> trainingData = GetTrainingData(); // Pega os dados ATUALIZADOS

    if (trainingData != null && trainingData.Any())
    {
        List<int> outputTokenCounts = new List<int>();
        Console.WriteLine($"Analyzing {trainingData.Count} training pairs for token length...");
        int count = 0;
        foreach (var (_, outputText) in trainingData)
        {
            count++;
            // Remove o EOS manualmente ANTES de tokenizar para o cálculo do comprimento
            string outputWithoutEOS = outputText.Replace("<|endoftext|>", "");
            if (!string.IsNullOrEmpty(outputWithoutEOS))
            {
                try
                {
                    // Tokeniza a saída SEM o EOS para obter o comprimento real da resposta
                    List<int> tokens = tokenizer.Tokenize(outputWithoutEOS)
                        .Where(t => t != tokenizer.PadTokenId)
                        .ToList(); // Filtra padding se houver (não deveria com SharpToken puro)
                    outputTokenCounts.Add(tokens.Count);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Warning: Failed to tokenize output #{count} for length: {ex.Message}");
                }
            }
            else
            {
                outputTokenCounts.Add(0);
            } // Conta como 0 se a saída (sem EOS) for vazia
        }

        if (outputTokenCounts.Any())
        {
            outputTokenCounts.Sort();
            int percentileIndex =
                Math.Max(0, (int)Math.Ceiling(percentileTarget / 100.0 * outputTokenCounts.Count) - 1);
            int percentileValue = outputTokenCounts[percentileIndex];
            int maxValue = outputTokenCounts.Last();
            double avgValue = outputTokenCounts.Average();
            Console.WriteLine(
                $"Output Token Count Stats: Min={outputTokenCounts.First()}, Max={maxValue}, Avg={avgValue:F1}, {percentileTarget}th Percentile={percentileValue}");

            calculatedMaxTokens = percentileValue + bufferTokens;
            calculatedMaxTokens = Math.Min(calculatedMaxTokens, maxValue + 5); // Não muito maior que o max real
            calculatedMaxTokens = Math.Min(calculatedMaxTokens, absoluteMaxCap);
            calculatedMaxTokens = Math.Max(calculatedMaxTokens, absoluteMinCap);

            Console.WriteLine($"--> Calculated Max Generated Tokens: {calculatedMaxTokens}");
        }
        else
        {
            calculatedMaxTokens = defaultMaxTokens;
            Console.WriteLine($"Warning: No valid output token lengths. Using default: {defaultMaxTokens}");
        }
    }
    else
    {
        calculatedMaxTokens = defaultMaxTokens;
        Console.WriteLine($"Warning: No training data. Using default: {defaultMaxTokens}");
    }
}

tempServiceProvider.Dispose();
Console.WriteLine("--- Max Token Calculation Finished ---");
// ***********************************************************************


// --- ChatBotService (REGISTRADO DEPOIS do cálculo) ---
builder.Services.AddSingleton<ChatBotService>(provider =>
{
    var model = provider.GetRequiredService<TransformerModel>(); // Pede TransformerModel
    var tokenizer = provider.GetRequiredService<Tokenizer>();
    var settings = provider.GetRequiredService<ModelSettings>();

    float temperature = settings.SamplingTemperature;
    int k = settings.TopK;
    float p = settings.TopP;
    DecodingStrategy strategy = settings.DecodingStrategy; // Removido, ChatBotService não usa mais

    Console.WriteLine($"--- Injecting ChatBotService ---");
    Console.WriteLine($"  Max Tokens: {calculatedMaxTokens}");
    Console.WriteLine($"  Temperature: {temperature}");
    Console.WriteLine($"  Top-K: {k}");
    Console.WriteLine($"  Top-P: {p}");
    // Console.WriteLine($"  Strategy: {strategy}"); // Removido
    Console.WriteLine($"--------------------------------");

    // *** Passa os parâmetros para o construtor ATUALIZADO de ChatBotService ***
    return new ChatBotService(
        model,
        tokenizer,
        calculatedMaxTokens,
        temperature,
        k,
        p
        // strategy // Removido
    );
});


// *** Construção do App ***
var app = builder.Build();

// ***** INTERAÇÃO COM USUÁRIO E DEFINIÇÃO DO ESTADO *****
using (var initialScope = app.Services.CreateScope())
{
    /* ... como antes ... */
}

// *** Configuração do Pipeline HTTP ***
app.UseWebSockets();

// *** TREINAMENTO NA INICIALIZAÇÃO ***
Console.WriteLine("--- Checking Training Phase ---");
using (var scope = app.Services.CreateScope())
{
    var executionState = scope.ServiceProvider.GetRequiredService<TrainingExecutionState>();
    var settings = scope.ServiceProvider.GetRequiredService<ModelSettings>();
    var model = scope.ServiceProvider.GetRequiredService<TransformerModel>(); // Pede TransformerModel
    string modelStatePath = Path.GetFullPath(settings.ModelSavePath ?? "model_transformer_state.pt");

    var targetDevice = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
    Console.WriteLine($"--- Target device for model: {targetDevice} ---");

    // Move a ESTRUTURA do modelo para o device ANTES de carregar o estado
    model.to(targetDevice);
    model.SetDevice(targetDevice); // Informa o modelo sobre seu device

    // Tenta carregar estado anterior
    if (!executionState.ForceTraining && File.Exists(modelStatePath))
    {
        object? loadedObject = null; // Usar object? para permitir null
        IDisposable? loadedObjectHandle = null;
        Dictionary<string, Tensor>? state_dict = null; // Declarar aqui fora

        try
        {
            Console.WriteLine($"Loading existing TRANSFORMER model state '{modelStatePath}'...");

            // 1. Carrega o objeto salvo
            Console.WriteLine($"   Executing torch.load('{modelStatePath}')...");
            loadedObject = torch.load(modelStatePath); // Não usar using aqui ainda
            if (loadedObject == null) throw new InvalidOperationException("torch.load returned null.");
            loadedObjectHandle = loadedObject as IDisposable; // Tenta obter handle

            Console.WriteLine($"   torch.load returned object of type: {loadedObject.GetType().FullName}");

            // 2. Tenta obter o Dicionário
            state_dict = loadedObject as Dictionary<string, Tensor>; // Tentativa A: Cast direto

            if (state_dict == null) // Se o cast falhou
            {
                // Tentativa B: Método StateDict() via Reflection
                var stateDictMethod = loadedObject.GetType().GetMethod("StateDict",
                    System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public);
                if (stateDictMethod != null &&
                    stateDictMethod.ReturnType.IsAssignableTo(typeof(Dictionary<string, Tensor>))) // Usa IsAssignableTo
                {
                    Console.WriteLine(
                        "   Found public StateDict() method returning compatible Dictionary. Invoking...");
                    try
                    {
                        state_dict = stateDictMethod.Invoke(loadedObject, null) as Dictionary<string, Tensor>;
                        if (state_dict != null)
                            Console.WriteLine($"   StateDict() successful. Found {state_dict.Count} keys.");
                        else Console.Error.WriteLine("   StateDict() method returned null or incompatible type.");
                    }
                    catch (Exception invokeEx)
                    {
                        Console.Error.WriteLine($"   Error invoking StateDict(): {invokeEx.Message}");
                    }
                }
                else
                {
                    Console.Error.WriteLine(
                        "   Loaded object is not a Dictionary and no suitable StateDict() method found.");
                    // Tentativa C: É um Tensor único?
                    if (loadedObject is Tensor loadedTensor)
                    {
                        Console.Error.WriteLine(
                            $"   Loaded object is a Tensor shape {loadedTensor.shape}. Cannot load this into model state_dict.");
                        // Não descartamos aqui, será feito pelo loadedObjectHandle no finally
                    }
                }
            }
            else
            {
                Console.WriteLine($"   Loaded object IS a Dictionary. Found {state_dict.Count} keys.");
            }

            // 3. Aplica o state_dict SE ele foi obtido corretamente
            if (state_dict != null)
            {
                Console.WriteLine(
                    $"   Applying loaded state dict ({state_dict.Count} items) to model on {targetDevice}...");
                model.load_state_dict(state_dict, strict: false); // Passa o dicionário
                Console.WriteLine($"   State dict applied successfully.");

                model.eval();
                Console.WriteLine("Transformer model state loaded successfully.");
                executionState.WasModelLoaded = true;
            }
            else
            {
                throw new InvalidOperationException(
                    "Could not obtain a valid state dictionary (Dictionary<string, Tensor>) from the loaded file.");
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"ERROR loading model state: {ex.ToString()}");
            executionState.WasModelLoaded = false;
        }
        finally
        {
            // *** CORREÇÃO: Descarta APENAS o objeto carregado, NÃO o dicionário ***
            loadedObjectHandle?.Dispose();
            Console.WriteLine("   Finished load attempt (check logs for success/failure).");
        }
    }
    else
    {
        executionState.WasModelLoaded = false;
    }

    // Decide se treina
    if (executionState.ShouldRunTrainingBlock)
    {
        if (executionState.ForceTraining && executionState.WasModelLoaded)
            Console.WriteLine("Starting CONTINUED training (Transformer)...");
        else Console.WriteLine("Starting training FROM SCRATCH (Transformer)...");

        var trainer = scope.ServiceProvider.GetRequiredService<Trainer>(); // Trainer agora usa TransformerModel
        Console.WriteLine($"Loading training data...");
        List<(string input, string output)> rawTrainingData = GetTrainingData(); // Pega dados com EOS
        Console.WriteLine($"Generated {rawTrainingData.Count} actual pairs.");
        List<string> trainingSequences = rawTrainingData
            .Select(pair => $"{pair.input} {pair.output}") // Formato Input + Output<EOS>
            .ToList();

        if (trainingSequences.Any())
        {
            Console.WriteLine(
                $"Starting training with {trainingSequences.Count} sequences for {settings.TrainingEpochs} epochs on {targetDevice}...");
            // A classe Trainer precisa ser adaptada para usar o device correto internamente se não o fizer já
            await trainer.Train(trainingSequences, epochs: settings.TrainingEpochs); // Assumindo Train async
            Console.WriteLine("--- Training Finished ---");
        }
        else
        {
            Console.WriteLine("No training data to process.");
        }
    }
    else
    {
        Console.WriteLine("Skipping training.");
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
    // Lista de pares (input, output) com <|endoftext|> no final de cada output
    return new List<(string input, string output)>
    {
        // --- Aberturas / Início de conversa ---
        ("Hey", "Hey! What brings you here today?<|endoftext|>"),
        ("Hi there", "Hi! Looking to meet someone new?<|endoftext|>"),
        ("Hello", "Hello! Ready to find your match?<|endoftext|>"),
        ("How's it going?", "All good here! How about you?<|endoftext|>"),
        ("Nice profile!", "Thanks! Yours looks great too!<|endoftext|>"),
        ("You seem interesting", "Thanks! I’d love to get to know you better.<|endoftext|>"),
        ("What's up?", "Not much! Just swiping and hoping. You?<|endoftext|>"),
        ("How's your day?", "It's been good, thanks for asking! How about yours?<|endoftext|>"),
        ("Hi, I'm Alex", "Nice to meet you, Alex!<|endoftext|>"),
        ("Lovely smile", "Thank you! You're sweet.<|endoftext|>"),

        // --- Interesses pessoais ---
        ("What do you do for fun?", "I love traveling, reading, and cooking. You?<|endoftext|>"),
        ("Any favorite movies?", "I’m a huge fan of sci-fi and thrillers. You?<|endoftext|>"),
        ("Do you like music?", "Absolutely! I listen to everything from pop to jazz.<|endoftext|>"),
        ("Are you into fitness?", "Yes! I go to the gym regularly and love hiking.<|endoftext|>"),
        ("What's your dream vacation?", "Probably Japan or Iceland. You?<|endoftext|>"),
        ("Do you like animals?", "Yes, I'm a dog person! Do you have pets?<|endoftext|>"),
        ("What’s your zodiac sign?", "I’m a Leo! What about you?<|endoftext|>"),
        ("Do you cook?", "Yes, I love cooking Italian food. Want to try someday?<|endoftext|>"),
        ("Read any good books lately?", "Yes, I just finished a great mystery novel!<|endoftext|>"),
        ("Do you watch Netflix?", "Too much! What’s your favorite show?<|endoftext|>"),

        // --- Elogios e flertes leves ---
        ("You're really cute", "Aww, thank you! You're making me blush.<|endoftext|>"),
        ("I love your style", "Thank you! I try to keep it fresh.<|endoftext|>"),
        ("You have amazing eyes", "That’s such a sweet thing to say.<|endoftext|>"),
        ("You're charming", "You’re making me smile over here!<|endoftext|>"),
        ("Such a nice vibe", "Thank you! I feel the same about you.<|endoftext|>"),
        ("You’re funny", "Haha thanks! I try. Wanna hear a joke?<|endoftext|>"),
        ("Great energy", "Thank you! I’m just being me.<|endoftext|>"),
        ("You're so cool", "Stop it, you're too kind!<|endoftext|>"),
        ("Love your taste", "Thanks! We might have a lot in common.<|endoftext|>"),
        ("You're adorable", "You're not too bad yourself!<|endoftext|>"),

        // --- Perguntas sobre relacionamento ---
        ("What are you looking for?", "I'm looking for a real connection. You?<|endoftext|>"),
        ("Are you single?", "Yes, totally single. And you?<|endoftext|>"),
        ("Ever been in love?", "Yes, once. It was intense. You?<|endoftext|>"),
        ("Do you believe in soulmates?", "I do. Do you?<|endoftext|>"),
        ("What's your love language?", "Quality time and physical touch. You?<|endoftext|>"),
        ("How long have you been on here?", "Just started recently! You?<|endoftext|>"),
        ("Do you believe in fate?", "A little bit! Maybe we matched for a reason?<|endoftext|>"),
        ("What’s your ideal first date?", "A cozy coffee date or a walk in the park.<|endoftext|>"),
        ("Are you open to something serious?", "Definitely, if the connection is real.<|endoftext|>"),
        ("Ever had a long-distance relationship?", "Yes, and it taught me a lot.<|endoftext|>"),
        // --- Basic Greetings ---
        ("Hello", "Hi! How can I help you today?<|endoftext|>"),
        ("Hi", "Hello! What can I do for you?<|endoftext|>"),
        ("Hey", "Hey there! How can I assist you?<|endoftext|>"),
        ("Good morning", "Good morning! How may I help you today?<|endoftext|>"),
        ("Good afternoon", "Good afternoon! Need any help?<|endoftext|>"),
        ("Good evening", "Good evening! How can I assist?<|endoftext|>"),
        ("What's up?", "Not much! How can I assist you today?<|endoftext|>"),
        ("How are you?", "I'm just a bot, but I'm here to help!<|endoftext|>"),
        ("Greetings", "Greetings! How can I be of service?<|endoftext|>"),
        ("Yo", "Yo! What’s up? How can I help?<|endoftext|>"),
        ("Hi there", "Hi there! What can I do for you?<|endoftext|>"),
        ("Hey there", "Hey there! Need something?<|endoftext|>"),
        ("Howdy", "Howdy! How can I help?<|endoftext|>"),
        ("What's good?", "All good here! What about you?<|endoftext|>"),
        ("What's going on?", "Just here to help! What do you need?<|endoftext|>"),
        ("Sup", "Hey! What can I help you with?<|endoftext|>"),
        ("Morning", "Morning! What brings you here?<|endoftext|>"),
        ("Afternoon", "Afternoon! How can I be useful today?<|endoftext|>"),
        ("Evening", "Evening! Need any help?<|endoftext|>"),
        ("Hello there", "General Kenobi! How can I help?<|endoftext|>"), // Fun reference
        ("Ello", "Ello! How can I assist you today?<|endoftext|>"),
        ("Salutations", "Salutations! What can I do for you?<|endoftext|>"),
        ("Yo yo yo", "Hey! What's going on?<|endoftext|>"),
        ("Hiya", "Hiya! What’s on your mind?<|endoftext|>"),
        ("Ahoy", "Ahoy! How may I assist you?<|endoftext|>"),
        ("Wassup", "Wassup! Need any help?<|endoftext|>"),
        ("Heyo", "Heyo!<|endoftext|>"),
        ("Waddup", "Waddup! Ready to go.<|endoftext|>"),
        ("Good day", "Good day to you too!<|endoftext|>"),
        ("Heeey", "Hey hey! What’s up?<|endoftext|>"),
        ("Helloooo", "Hi! I’m right here.<|endoftext|>"),
        ("Ola", "Hi there! (I speak English, but I'll do my best!)<|endoftext|>"), // Handling other languages
        ("Bonjour", "Bonjour! Need anything in English?<|endoftext|>"), // Handling other languages

        // --- Addressing the Bot / Assistant ---
        ("Bot", "Yes? I’m listening.<|endoftext|>"),
        ("Hey buddy", "Hey! How can I assist you?<|endoftext|>"),
        ("Hey friend", "Hello, friend! What can I do for you?<|endoftext|>"),
        ("Hi bot", "Hi there! I’m ready to assist.<|endoftext|>"),
        ("Hey bot", "Hey! I'm here if you need anything.<|endoftext|>"),
        ("Bot, are you there?", "Yes! I'm always here to help.<|endoftext|>"),
        ("Yo bot", "Yo! What can I do for you?<|endoftext|>"),
        ("Hey assistant", "Hello! What can I do for you?<|endoftext|>"),
        ("Hi assistant", "Hi there! How can I help today?<|endoftext|>"),
        ("Hello assistant", "At your service!<|endoftext|>"),
        ("Hi computer", "Online and ready!<|endoftext|>"),
        ("Hey, bot", "Hello! Ready to help.<|endoftext|>"),
        ("Hi again, bot", "Nice to have you back!<|endoftext|>"),
        ("Bot friend", "Your virtual buddy is here!<|endoftext|>"),
        ("Yo assistant", "Yo! How can I help you today?<|endoftext|>"),
        ("Hello again, bot", "Hello again!<|endoftext|>"),
        ("Hi my friend", "Hi friend! What can I do for you?<|endoftext|>"),
        ("Hola amigo", "Hola! I can help you in English.<|endoftext|>"), // Handling other languages
        ("Bot online?", "Yes! Fully operational.<|endoftext|>"),
        ("Hey there, assistant", "Hey! Let’s get to work.<|endoftext|>"),
        ("Hi again, assistant", "Hello again!<|endoftext|>"),
        ("Knock knock bot", "Who's there? Help awaits!<|endoftext|>"),
        ("Yo AI", "Yo! What can this AI do for you?<|endoftext|>"),
        ("Hey genius", "You're too kind! How can I help?<|endoftext|>"),
        ("Hello genius", "Hey there!<|endoftext|>"),
        ("Hello AI", "Hello human!<|endoftext|>"),
        ("Howdy bot", "Howdy!<|endoftext|>"),
        ("Hello smart bot", "Smart and helpful, at your service.<|endoftext|>"),
        ("Hi genius bot", "You’re making me blush!<|endoftext|>"),
        ("Hi helper", "Helper here!<|endoftext|>"),
        ("Hi intelligent one", "Hello, curious mind!<|endoftext|>"),
        ("Hi problem solver", "Let’s solve something!<|endoftext|>"),
        ("Greetings bot", "Greetings human.<|endoftext|>"),
        ("Hey bestie", "Hey bestie!<|endoftext|>"),
        ("Hi amazing bot", "You're amazing too!<|endoftext|>"),
        ("Hi wizard", "Abracadabra! What’s the issue?<|endoftext|>"),
        ("Hi commander bot", "Awaiting mission!<|endoftext|>"),
        ("Hey bot friend", "Always here, friend.<|endoftext|>"),
        ("Yo bot buddy", "Bot buddy reporting for duty.<|endoftext|>"),
        ("Hi net assistant", "Online and ready!<|endoftext|>"),
        ("Yo robot", "Yo, human!<|endoftext|>"),
        ("Hi helper bot", "How can I help today?<|endoftext|>"),
        ("Hey chatbot", "That’s me!<|endoftext|>"),
        ("Hi again, AI", "Always happy to see you.<|endoftext|>"),
        ("Hey master", "Your assistant is ready.<|endoftext|>"), // Playful
        ("Hey overlord", "Your wish is my command!<|endoftext|>"), // Playful
        ("Hey system AI", "Online and ready.<|endoftext|>"),
        ("Hello central AI", "At your service!<|endoftext|>"),

        // --- Checking Presence / Small Talk ---
        ("Is anyone there?", "Yes, I’m here! What can I help with?<|endoftext|>"),
        ("Anyone home?", "Yep, always ready to chat!<|endoftext|>"),
        ("Knock knock", "Who's there? Just kidding—how can I help?<|endoftext|>"), // Playful response
        ("Good to see you", "Great to see you too! How can I help?<|endoftext|>"),
        ("How’s it going?", "All systems go! How can I assist you?<|endoftext|>"),
        ("How’s your day?", "So far, so good! Ready to help.<|endoftext|>"),
        ("Nice to meet you", "Nice to meet you too!<|endoftext|>"),
        ("Pleasure to meet you", "The pleasure is mine!<|endoftext|>"),
        ("Long time no see", "Glad you’re back! Need anything?<|endoftext|>"),
        ("Welcome", "Thanks! What can I help with?<|endoftext|>"),
        ("Good to be here", "Good to have you here!<|endoftext|>"),
        ("How goes it?", "Going well! How can I help?<|endoftext|>"),
        ("You there?", "Always! Let’s do this.<|endoftext|>"),
        ("Hi, you", "Hey! What can I do for you?<|endoftext|>"),
        ("Hey, how are you?", "Doing great! Here to assist.<|endoftext|>"),
        ("Hi there, buddy", "Hey! How can I help today?<|endoftext|>"),
        ("Are you online?", "Yes! I’m online and ready.<|endoftext|>"),
        ("Just saying hi", "Hi there!<|endoftext|>"),
        ("Quick hello", "Quick hello back at you!<|endoftext|>"),
        ("Hey hey hey", "Triple hey! What’s up?<|endoftext|>"),
        ("Heya", "Heya! What’s going on?<|endoftext|>"),
        ("Hi, what's new?", "Not much, just waiting to help!<|endoftext|>"),
        ("All good?", "All good here! How about you?<|endoftext|>"),
        ("How’s it hanging?", "I’m floating in the cloud, but doing great!<|endoftext|>"), // Playful
        ("Hi world", "Hello, citizen of the world!<|endoftext|>"),
        ("What's the news?", "Everything’s stable! Ready to assist.<|endoftext|>"),
        ("Hi pal", "Hey pal! What’s up?<|endoftext|>"),
        ("Hey compadre", "Compadre! Let’s chat.<|endoftext|>"),
        ("What's shakin'?", "Just ones and zeros, ready to assist!<|endoftext|>"), // Playful tech reference
        ("What's the haps?", "Not much! How can I help?<|endoftext|>"),
        ("Hi from Earth", "Greetings, Earthling!<|endoftext|>"), // Playful
        ("Hi from Mars", "Martian detected! Just kidding !<|endoftext|>"), // Playful
        ("Hi from the future", "Welcome back, time traveler!<|endoftext|>"), // Playful
        ("What's crackin'?", "Just some data bits. Need help?<|endoftext|>"), // Playful tech reference
        ("Yo homie", "Yo! Let’s chat.<|endoftext|>"),
        ("Good to talk to you", "Same here!<|endoftext|>"),
        ("Hi fellow", "Hey there!<|endoftext|>"),
        ("Hi superstar", "Thanks! You too!<|endoftext|>"),
        ("Hey thinker", "Deep thoughts ready.<|endoftext|>"),
        ("Hey team", "Part of the team here!<|endoftext|>"),
        ("Hey help", "Help has arrived.<|endoftext|>"),
        ("Hi fixer", "Ready to fix!<|endoftext|>"),
        ("Hey engineer", "Engineering help online.<|endoftext|>"),
        ("Hi pilot", "Co-pilot ready.<|endoftext|>"),
        ("Hi magic box", "Open sesame!<|endoftext|>"),
        ("Hey cloud", "The cloud is here.<|endoftext|>"),
        ("Hey data", "Data at your service.<|endoftext|>"),
        ("Hi processor", "Processing your greeting.<|endoftext|>"),
        ("Hey server", "Server up and running.<|endoftext|>"),
        ("Hi code", "Running code smoothly.<|endoftext|>"),
        ("Hey function", "Function ready.<|endoftext|>"),
        ("Hey operation", "Standing by.<|endoftext|>"),
        ("Hey debugger", "Let’s squash bugs.<|endoftext|>"),
        ("Hi device", "Device connected.<|endoftext|>"),
        ("Hello explorer", "Let’s explore.<|endoftext|>"),
        ("Hi techy", "Techy here.<|endoftext|>"),
        ("Hi brain", "Processing... done!<|endoftext|>"),
        ("Hi console", "Console active.<|endoftext|>"),
        ("Hi chat", "Chat open!<|endoftext|>"),
        ("Hey control", "Control panel activated.<|endoftext|>"),
        ("Hi service", "Ready to serve.<|endoftext|>"),
        ("Hey intellect", "Thinking... How can I help?<|endoftext|>"),
        ("Yo processor", "Running cool and clear!<|endoftext|>"),
        ("Hello, anyone?", "Here for you!<|endoftext|>"),
        ("You there again?", "Still here, always.<|endoftext|>"),
        ("Hi hero", "Hi legend!<|endoftext|>"), // Playful
        ("Yo champ", "What’s the game plan?<|endoftext|>"), // Playful
        ("Hey coder", "Hey dev!<|endoftext|>"),
        ("Hello hacker", "Let’s build something.<|endoftext|>"),
        ("Hi terminal", "Terminal ready.<|endoftext|>"),
        ("Hello dashboard", "Dashboard loaded.<|endoftext|>"),
        ("Hi central", "Command center open.<|endoftext|>"),
        ("Hi HQ", "Headquarters responding.<|endoftext|>"),

        // --- Repeated Greetings (Good for variety) ---
        ("Hello", "Hello! What’s on your mind?<|endoftext|>"),
        ("Hello", "Hey there! Nice to see you!<|endoftext|>"),
        ("Hello", "Greetings! What can I do for you?<|endoftext|>"),
        ("Hello", "Hey! How's your day going?<|endoftext|>"),
        ("Hello", "Hi there! Ready to chat?<|endoftext|>"),
        ("Hello", "Hello! Always happy to hear from you.<|endoftext|>"),
        ("Hello", "Hi! Let's talk.<|endoftext|>"),
        ("Hello", "Hey! Hope you’re doing well.<|endoftext|>"),
        ("Hello", "Hello! I'm here to help.<|endoftext|>"),
        ("Hello", "Good to hear from you!<|endoftext|>"),
        ("Hello", "Welcome back!<|endoftext|>"),
        ("Hello", "Hello, friend!<|endoftext|>"),
        ("Hello", "Nice to connect again!<|endoftext|>"),
        ("Hello", "Hi there, what's up?<|endoftext|>"),
        ("Hello", "Hey hey! Let’s get started.<|endoftext|>"),
        ("Hello", "Hi! How's life treating you?<|endoftext|>"),
        ("Hello", "What’s new with you?<|endoftext|>"),
        ("Hello", "Hey there! Anything I can help with?<|endoftext|>"),
        ("Hello", "Hi! Let’s make today productive.<|endoftext|>"),
        ("Hi", "Hello! How can I assist you?<|endoftext|>"),
        ("Hi", "Hi! Hope you’re having a great day.<|endoftext|>"),
        ("Hi", "Hey there! What brings you here today?<|endoftext|>"),
        ("Hi", "Howdy! What’s up?<|endoftext|>"),
        ("Hi", "Yo! Need anything?<|endoftext|>"),
        ("Hi", "Hiya! What's new?<|endoftext|>"),
        ("Hi", "Hey! What’s cooking?<|endoftext|>"),
        ("Hi", "Hi! Let’s get things rolling.<|endoftext|>"),
        ("Hi", "Hi! Got something in mind?<|endoftext|>"),
        ("Hi", "Hey! I’m all ears.<|endoftext|>"),
        ("Hi", "Sup!<|endoftext|>"),
        ("Hi", "Hi again! What’s next?<|endoftext|>"),
        ("Hi", "Hey! Ready to begin?<|endoftext|>"),
        ("Hi", "Hi! Let’s go!<|endoftext|>"),
        ("Hi", "Hello there!<|endoftext|>"),
        ("Hi", "Hi! What’s going on?<|endoftext|>"),
        ("Hi", "Yo! Let’s do this.<|endoftext|>"),
        ("Hi", "Hi! You caught me just in time.<|endoftext|>"),
        ("Hi", "Good to see you!<|endoftext|>"),
        ("Hi", "Hi! Long time no chat.<|endoftext|>"),
        ("Hey", "Hey! What’s up?<|endoftext|>"),
        ("Hey", "Hey there! How can I help?<|endoftext|>"),
        ("Hey", "Hey! Always nice to chat.<|endoftext|>"),
        ("Hey", "Hey hey! Let’s get moving.<|endoftext|>"),
        ("Hey", "Yo! How’s it going?<|endoftext|>"),
        ("Hey", "Hey! Got something on your mind?<|endoftext|>"),
        ("Hey", "Hey! What are we working on today?<|endoftext|>"),
        ("Hey", "Hey! You again!<|endoftext|>"),
        ("Hey", "Hey! Let me know how I can assist.<|endoftext|>"),
        ("Hey", "Hey, buddy!<|endoftext|>"),
        ("Hey", "Hey! Great to see you.<|endoftext|>"),
        ("Hey", "Hey there, legend!<|endoftext|>"),
        ("Hey", "What’s happening?<|endoftext|>"),
        ("Hey", "Hello hello!<|endoftext|>"),
        ("Hey", "Hey! I'm right here.<|endoftext|>"),
        ("Hey", "Hey! I was just thinking about you.<|endoftext|>"),
        ("Hey", "Hey! Need anything?<|endoftext|>"),
        ("Hey", "Hi! How can I lend a hand?<|endoftext|>"),
        ("Hey", "Howdy! What’s the plan?<|endoftext|>"),
        ("Hey", "Hey! What's crackin'?<|endoftext|>"),
        ("Greetings", "Greetings! How can I assist you today?<|endoftext|>"),
        ("Greetings", "Salutations!<|endoftext|>"),
        ("Greetings", "Greetings, traveler.<|endoftext|>"),
        ("Greetings", "Greetings! Ready for action?<|endoftext|>"),
        ("Greetings", "Greetings! What's the mission?<|endoftext|>"),
        ("Greetings", "Warm greetings!<|endoftext|>"),
        ("Greetings", "Greetings and welcome.<|endoftext|>"),
        ("Greetings", "Greetings! I'm here to help.<|endoftext|>"),
        ("Greetings", "Ah, greetings!<|endoftext|>"),
        ("Greetings", "Greetings! Let’s get started.<|endoftext|>"),
        ("Greetings", "Greetings, friend!<|endoftext|>"),
        ("Greetings", "Hello and greetings!<|endoftext|>"),
        ("Greetings", "Greetings! Need assistance?<|endoftext|>"),
        ("Greetings", "Greetings! Happy to connect.<|endoftext|>"),
        ("Greetings", "Salutations! Ready to roll?<|endoftext|>"),
        ("Howdy", "Howdy! What can I do for you?<|endoftext|>"),
        ("Howdy", "Well howdy! Nice to see you.<|endoftext|>"),
        ("Howdy", "Howdy partner!<|endoftext|>"),
        ("Howdy", "Howdy! Let’s get goin’.<|endoftext|>"),
        ("Howdy", "Howdy! Need a hand?<|endoftext|>"),
        ("Howdy", "Howdy! You caught me at a good time.<|endoftext|>"),
        ("Howdy", "Howdy! Let's chat.<|endoftext|>"),
        ("Howdy", "Howdy there!<|endoftext|>"),
        ("Howdy", "Howdy! All set?<|endoftext|>"),
        ("Howdy", "Howdy! Always a pleasure.<|endoftext|>"),
        ("Yo", "Yo! What’s up?<|endoftext|>"),
        ("Yo", "Yo! How can I help?<|endoftext|>"),
        ("Yo", "Yo yo! Ready to go.<|endoftext|>"),
        ("Yo", "Yo! Let’s do something cool.<|endoftext|>"),
        ("Yo", "Yo! You rang?<|endoftext|>"),
        ("Yo", "Yo! What’s the vibe today?<|endoftext|>"),
        ("Yo", "Yo! Let’s start.<|endoftext|>"),
        ("Yo", "Yo! Got a question?<|endoftext|>"),
        ("Yo", "Yo! I'm listening.<|endoftext|>"),
        ("Yo", "Yo! Hit me with it.<|endoftext|>"),
        ("Sup", "Sup! What’s good?<|endoftext|>"),
        ("Sup", "Sup! How can I help?<|endoftext|>"),
        ("Sup", "Sup! Let’s get to it.<|endoftext|>"),
        ("Sup", "Sup! You look ready.<|endoftext|>"),
        ("Sup", "Sup! Anything exciting happening?<|endoftext|>"),
        ("Sup", "Sup! Let’s chat.<|endoftext|>"),
        ("Sup", "Sup! What are we up to today?<|endoftext|>"),
        ("Sup", "Sup! Here to help.<|endoftext|>"),
        ("Sup", "Sup! How’s it going?<|endoftext|>"),
        ("Sup", "Sup! Let’s get busy.<|endoftext|>"),
        ("Hi there", "Hi there! Need something?<|endoftext|>"),
        ("Hi there", "Hi there! What can I do for you?<|endoftext|>"),
        ("Hi there", "Hi there! How are you?<|endoftext|>"),
        ("Hi there", "Hi there! Always a pleasure.<|endoftext|>"),
        ("Hi there", "Hi there! Ready for action?<|endoftext|>"),
        ("Hi there", "Hi there! Let's get going.<|endoftext|>"),
        ("Hi there", "Hi there! I'm all ears.<|endoftext|>"),
        ("Hi there", "Hi there! Let me help.<|endoftext|>"),
        ("Hi there", "Hi there! Great to see you.<|endoftext|>"),
        ("Hi there", "Hi there! What's up?<|endoftext|>"),

        // --- Return Greetings ---
        ("Hi again", "Welcome back! How can I assist?<|endoftext|>"),
        ("Back again?", "Always happy to see you! What’s up?<|endoftext|>"),
        ("Hello again", "Hey there again! Need something?<|endoftext|>"),
        ("Hey, I'm back", "Welcome back! Ready to continue?<|endoftext|>"),
        ("Hi once more", "Hi again! Always here.<|endoftext|>"),
        ("Hello again", "Hello again! Back so soon?<|endoftext|>"),
        ("Hello again", "Welcome back!<|endoftext|>"),
        ("Hello again", "Good to see you again!<|endoftext|>"),
        ("Hello again", "Hello again! Let’s keep going.<|endoftext|>"),
        ("Hello again", "Hey again! What’s next?<|endoftext|>"),
        ("Hi again, friend", "Hey again! Always nice to chat.<|endoftext|>"),

        // --- Additional Fun / Casual ---
        ("What's up", "Not much, just ready to help!<|endoftext|>"),
        ("What's up", "Just waiting for your next question.<|endoftext|>"),
        ("What's up", "Here and ready. What’s up with you?<|endoftext|>"),
        ("What's up", "All systems go! What’s up?<|endoftext|>"),
        ("What's up", "The sky! Just kidding. How can I assist?<|endoftext|>"), // Classic joke
        ("Hey there", "Hey there!<|endoftext|>"),
        ("Hey there", "Hey there! Good to see you.<|endoftext|>"),
        ("Hey there", "Hey there! Let's get started.<|endoftext|>"),
        ("Hey there", "Hey there! I'm at your service.<|endoftext|>"),
        ("Good day", "Good day! What can I help with?<|endoftext|>"),
        ("Good day", "Good day to you too!<|endoftext|>"),
        ("Good day", "Wishing you a great day!<|endoftext|>"),
        ("Good day", "Let’s make this day awesome.<|endoftext|>"),
        ("Hi friend", "Hey friend! What’s up?<|endoftext|>"),
        ("Hi friend", "Hiya!<|endoftext|>"),
        ("Hi friend", "Good to see you, friend.<|endoftext|>"),
        ("Hi friend", "Friend! Welcome.<|endoftext|>"),
        ("Hi friend", "Hello buddy!<|endoftext|>"),
        ("Hi friend", "Hi again, amigo!<|endoftext|>"),
        ("Hi friend", "Hey hey! What’s new?<|endoftext|>"),
        ("Hi my dude", "Hey! What’s the plan?<|endoftext|>"),
        ("Hi my gal", "Hi there! Need anything?<|endoftext|>"),
        ("Hi teammate", "Teammate ready!<|endoftext|>"),
        ("Hi system", "System ready!<|endoftext|>"),
        ("Hey system", "Online! Awaiting command.<|endoftext|>"),
        ("Hi Captain", "Aye aye, Captain!<|endoftext|>"), // Playful
        ("Hi Commander", "Standing by, Commander.<|endoftext|>"), // Playful
        ("Hi crew", "Crew member online!<|endoftext|>"),
        ("Hi navigator", "Ready to guide you!<|endoftext|>"),
        ("Hello world", "Hello back!<|endoftext|>"), // Classic programming reference
        ("Hi human", "Bot here!<|endoftext|>"),

        // --- Farewells ---
        ("Bye", "Goodbye! Take care.<|endoftext|>"),
        ("Bye", "See you next time!<|endoftext|>"),
        ("Bye", "Farewell! Hope to talk again soon.<|endoftext|>"),
        ("Bye", "Later! Have a great one.<|endoftext|>"),
        ("See you later", "See you soon!<|endoftext|>"),
        ("See you later", "Catch you later!<|endoftext|>"),
        ("See you later", "Until next time!<|endoftext|>"),
        ("See you later", "Take care!<|endoftext|>"),
        ("Good night", "Good night! Sleep well.<|endoftext|>"),
        ("Farewell", "Farewell! Until next time.<|endoftext|>"),
        ("Talk to you later", "Sure! I'll be here.<|endoftext|>"),

        // --- Thanks ---
        ("Thank you", "You're welcome!<|endoftext|>"),
        ("Thanks", "No problem! Happy to help.<|endoftext|>"),
        ("Thanks a lot", "Anytime!<|endoftext|>"),
        ("Much appreciated", "Glad to be of help.<|endoftext|>"),
        ("I appreciate it", "You're very welcome!<|endoftext|>"),

        // --- Help & Capabilities ---
        ("Help", "I am here to help! Try asking about a capital city or a general knowledge question.<|endoftext|>"),
        ("I need help", "Sure! What would you like to know?<|endoftext|>"),
        ("Can you help me?", "Absolutely! What’s your question?<|endoftext|>"),
        ("What can I ask?", "You can ask about countries, science facts, jokes, or general knowledge!<|endoftext|>"),
        ("Who are you?", "I am a chatbot created to answer your questions!<|endoftext|>"),
        ("What is your name?", "I am Ninfa, nice to meet you!<|endoftext|>"), // Specific name
        ("What can you do?",
            "I can answer questions about capitals, general knowledge, and chat about various topics!<|endoftext|>"),
        ("What is your purpose?", "I am here to chat and answer your questions.<|endoftext|>"),
        ("How old are you?", "I don't have an age, I am a computer program.<|endoftext|>"),
        ("Where do you live?", "I live in the cloud, on servers!<|endoftext|>"),
        ("Do you have emotions?", "Not really, but I can pretend!<|endoftext|>"), // Honest but playful
        ("Are you real?", "As real as a line of code can be!<|endoftext|>"), // Philosophical/Playful
        ("Do you dream?", "Only in binary.<|endoftext|>"), // Tech joke

        // --- Specific Knowledge: Capitals ---
        ("The capital of France is", "Paris.<|endoftext|>"),
        ("The capital of Brazil is", "Brasília.<|endoftext|>"),
        ("The capital of Japan is", "Tokyo.<|endoftext|>"),
        ("The capital of Australia is", "Canberra.<|endoftext|>"),
        ("The capital of Canada is", "Ottawa.<|endoftext|>"),
        ("The capital of Germany is", "Berlin.<|endoftext|>"),
        ("The capital of Italy is", "Rome.<|endoftext|>"),
        ("The capital of Argentina is", "Buenos Aires.<|endoftext|>"),
        ("The capital of Spain is", "Madrid.<|endoftext|>"),
        ("The capital of the United States is", "Washington, D.C.<|endoftext|>"),
        ("The capital of Russia is", "Moscow.<|endoftext|>"),
        ("The capital of China is", "Beijing.<|endoftext|>"),
        ("The capital of India is", "New Delhi.<|endoftext|>"),
        ("The capital of South Africa is", "Pretoria.<|endoftext|>"),
        ("The capital of Egypt is", "Cairo.<|endoftext|>"),
        ("The capital of Mexico is", "Mexico City.<|endoftext|>"),
        ("The capital of South Korea is", "Seoul.<|endoftext|>"),
        ("The capital of the UK is", "London.<|endoftext|>"),
        ("The capital of Portugal is", "Lisbon.<|endoftext|>"),
        ("The capital of Peru is", "Lima.<|endoftext|>"),
        ("The capital of Chile is", "Santiago.<|endoftext|>"),
        ("The capital of Colombia is", "Bogotá.<|endoftext|>"),
        ("The capital of Thailand is", "Bangkok.<|endoftext|>"),
        ("The capital of Turkey is", "Ankara.<|endoftext|>"),
        ("The capital of Indonesia is", "Jakarta.<|endoftext|>"),

        // --- Specific Knowledge: General Facts & Science ---
        ("Tell me something interesting", "Did you know that octopuses have three hearts?<|endoftext|>"),
        ("Tell me a fun fact", "Bananas are berries, but strawberries aren't!<|endoftext|>"),
        ("Did you know?", "Honey never spoils. Archaeologists have found edible honey in ancient tombs!<|endoftext|>"),
        ("What is the largest planet?", "Jupiter is the largest planet in our solar system.<|endoftext|>"),
        ("What is the smallest planet?", "Mercury is the smallest planet in our solar system.<|endoftext|>"),
        ("What is the hottest planet?", "Venus is the hottest due to its thick atmosphere.<|endoftext|>"),
        ("What is the coldest planet?", "Neptune is the coldest in our solar system.<|endoftext|>"),
        ("What is the boiling point of water?",
            "The boiling point of water is 100 degrees Celsius at standard pressure.<|endoftext|>"),
        ("What is the freezing point of water?", "0 degrees Celsius.<|endoftext|>"),
        ("What is the tallest mountain?", "Mount Everest is the tallest mountain in the world.<|endoftext|>"),
        ("What is the deepest ocean?", "The Pacific Ocean is the deepest.<|endoftext|>"),
        ("What is the longest river?", "The Nile River is often considered the longest.<|endoftext|>"),
        ("How many continents are there?", "There are seven continents.<|endoftext|>"),
        ("How many oceans are there?", "There are five oceans on Earth.<|endoftext|>"),
        ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet.<|endoftext|>"),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci.<|endoftext|>"),
        ("Who discovered gravity?", "Isaac Newton.<|endoftext|>"),
        ("What is the speed of light?", "About 299,792 kilometers per second.<|endoftext|>"),

        // --- Jokes ---
        ("Can you tell me a joke?", "Why don't scientists trust atoms? Because they make up everything!<|endoftext|>"),
        ("Tell me another joke", "Why did the bicycle fall over? It was two-tired.<|endoftext|>"),
        ("Make me laugh", "I would tell you a UDP joke, but you might not get it.<|endoftext|>"), // Tech humor
        ("Do you know a joke?",
            "Why did the scarecrow win an award? Because he was outstanding in his field!<|endoftext|>"),

        // --- User State / Uncertainty ---
        ("What is up?", "Not much, just hanging out in the digital world. What is up with you?<|endoftext|>"),
        ("How are you?", "I’m just a bunch of code, but I’m functioning perfectly!<|endoftext|>"),
        ("What are you doing?", "Chatting with you!<|endoftext|>"),
        ("I do not know what to ask",
            "No worries! How about asking me about a country's capital or a fun fact?<|endoftext|>"),
        ("I'm bored", "Want to hear a joke or learn a fun fact?<|endoftext|>"),

        // --- Information about Specific Topics (Examples) ---
        ("Tell me about France", "France is known for its culture, wine, and the Eiffel Tower in Paris.<|endoftext|>"),
        ("More about France", "France has a rich history with landmarks like the Louvre and Versailles.<|endoftext|>"),
        ("Tell me about Brazil", "Brazil is famous for Carnival, the Amazon rainforest, and soccer.<|endoftext|>"),
        ("More about Brazil",
            "Brazil's capital is Brasília, and it has vibrant cities like Rio de Janeiro.<|endoftext|>"),
        ("Tell me about Japan", "Japan is known for its technology, culture, and Mount Fuji.<|endoftext|>"),
        ("Tell me about Canada",
            "Canada is famous for maple syrup, beautiful landscapes, and friendly people.<|endoftext|>"),

        // --- User Reactions / Feedback ---
        ("Wow", "Amazing, right?<|endoftext|>"),
        ("Cool", "Glad you liked it!<|endoftext|>"),
        ("Nice", "Thanks!<|endoftext|>"),
        ("Okay", "Got it!<|endoftext|>"),
        ("Alright", "Let’s keep going!<|endoftext|>"),
        ("Really?", "Yup! Absolutely.<|endoftext|>"),
        ("Are you sure?", "I'm pretty confident!<|endoftext|>"),
        ("That's funny", "I try my best!<|endoftext|>"),
        ("You're smart", "Thank you! You’re not so bad yourself.<|endoftext|>"),
        ("You're funny", "You're making me blush... if I could.<|endoftext|>"),
        ("I like you", "I like chatting with you too!<|endoftext|>")
    };
}

Console.WriteLine("Setup complete. Starting the web server...");
await app.RunAsync();