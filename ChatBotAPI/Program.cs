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
int absoluteMaxCap = 128; // Ajuste conforme necessário
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
            Console.WriteLine($"Carregando estado existente do modelo TRANSFORMER '{modelStatePath}'...");

            // 1. Carrega o objeto salvo
            Console.WriteLine($"   Executando torch.load('{modelStatePath}')...");
            loadedObject = torch.load(modelStatePath); // Não usar using aqui ainda
            if (loadedObject == null) throw new InvalidOperationException("torch.load retornou null.");
            loadedObjectHandle = loadedObject as IDisposable; // Tenta obter handle para descarte posterior

            Console.WriteLine($"   torch.load retornou objeto do tipo: {loadedObject.GetType().FullName}");

            // 2. Tenta obter o Dicionário (state_dict)
            state_dict = loadedObject as Dictionary<string, Tensor>; // Tentativa A: Cast direto

            if (state_dict == null) // Se o cast direto falhou
            {
                Console.WriteLine(
                    "   Cast direto para Dictionary<string, Tensor> falhou. Tentando reflexão para StateDict()...");
                // Tentativa B: Método StateDict() via Reflexão
                var stateDictMethod = loadedObject.GetType().GetMethod("StateDict",
                    System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public);

                // Verifica se o método existe E se seu tipo de retorno é assignável a Dictionary<string, Tensor>
                if (stateDictMethod != null &&
                    stateDictMethod.ReturnType.IsAssignableTo(typeof(Dictionary<string, Tensor>)))
                {
                    Console.WriteLine(
                        "   Encontrado método público StateDict() retornando Dictionary compatível. Invocando...");
                    try
                    {
                        state_dict = stateDictMethod.Invoke(loadedObject, null) as Dictionary<string, Tensor>;
                        if (state_dict != null)
                            Console.WriteLine(
                                $"   StateDict() invocado com sucesso. Encontradas {state_dict.Count} chaves.");
                        else
                            Console.Error.WriteLine(
                                "   Método StateDict() retornou null ou um tipo incompatível após invocação.");
                    }
                    catch (Exception invokeEx)
                    {
                        Console.Error.WriteLine($"   Erro ao invocar StateDict() via reflexão: {invokeEx.Message}");
                        // state_dict permanecerá null
                    }
                }
                else
                {
                    Console.Error.WriteLine(
                        "   Objeto carregado não é um Dictionary e nenhum método StateDict() adequado encontrado via reflexão.");
                    // Tentativa C: É um Tensor único? Ou outro tipo inesperado?
                    if (loadedObject is Tensor loadedTensor)
                    {
                        Console.Error.WriteLine(
                            $"   Objeto carregado é um único Tensor com shape {loadedTensor.shape}. Não é possível carregar isso no state_dict do modelo.");
                        // Não precisamos descartar loadedTensor aqui, será feito pelo loadedObjectHandle no finally
                    }
                    else
                    {
                        Console.Error.WriteLine(
                            $"   Objeto carregado é de um tipo inesperado: {loadedObject.GetType().FullName}");
                    }
                }
            }
            else
            {
                Console.WriteLine(
                    $"   Objeto carregado É um Dictionary (cast direto bem-sucedido). Encontradas {state_dict.Count} chaves.");
            }

            // 3. Aplica o state_dict SE ele foi obtido corretamente
            if (state_dict != null)
            {
                // Obtém o dispositivo alvo do modelo (já movido anteriormente)
                var currentModelDevice = model.device; // Ou use targetDevice diretamente
                Console.WriteLine(
                    $"   Aplicando state_dict carregado ({state_dict.Count} itens) ao modelo no dispositivo {currentModelDevice}...");

                // Garante que o modelo esteja no dispositivo correto ANTES de carregar o estado
                // Esta linha pode ser redundante se já foi movido antes, mas garante
                model.to(currentModelDevice);

                model.load_state_dict(state_dict, strict: false); // Passa o dicionário extraído
                Console.WriteLine($"   State dict aplicado com sucesso.");

                model.eval(); // Coloca em modo de avaliação após carregar
                Console.WriteLine("Estado do modelo Transformer carregado com sucesso.");
                executionState.WasModelLoaded = true;
            }
            else
            {
                // Se não conseguiu obter o state_dict por nenhum método
                throw new InvalidOperationException(
                    "Não foi possível obter um dicionário de estado (Dictionary<string, Tensor>) válido do arquivo carregado. Verifique os logs para detalhes sobre o tipo do objeto carregado.");
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"ERRO ao carregar o estado do modelo: {ex.ToString()}");
            Console.Error.WriteLine($"Carregamento do modelo FALHOU. O treinamento prosseguirá do zero se habilitado.");
            executionState.WasModelLoaded = false;
            // Não relança a exceção aqui para permitir que o programa continue e treine do zero se necessário
        }
        finally
        {
            // *** CORREÇÃO IMPORTANTE: Descarta APENAS o objeto carregado via torch.load ***
            // O 'state_dict' é apenas uma referência ao dicionário DENTRO do objeto carregado
            // ou um dicionário padrão se o cast direto funcionou. Não descarte 'state_dict'.
            if (loadedObjectHandle != null)
            {
                Console.WriteLine($"   Descartando handle do objeto carregado ({loadedObject?.GetType().Name})...");
                loadedObjectHandle.Dispose();
            }
            else if (loadedObject != null)
            {
                Console.WriteLine(
                    $"   Objeto carregado ({loadedObject.GetType().Name}) não implementou IDisposable diretamente. Assumindo recursos gerenciados.");
            }

            Console.WriteLine(
                "   Tentativa de carregamento do modelo finalizada (verifique os logs acima para sucesso/falha).");
        }
    }
    else // Se o arquivo não existe ou ForceTraining está ativo
    {
        if (executionState.ForceTraining) Console.WriteLine("ForceTraining habilitado.");
        if (!File.Exists(modelStatePath))
            Console.WriteLine($"Arquivo de estado do modelo '{modelStatePath}' não encontrado.");
        executionState.WasModelLoaded = false;
    }

    // Decide se treina
    if (executionState.ShouldRunTrainingBlock)
    {
        
        if (executionState.ForceTraining && executionState.WasModelLoaded)
            Console.WriteLine("Starting CONTINUED training (Transformer)...");
        else Console.WriteLine("Starting training FROM SCRATCH (Transformer)...");
        Console.WriteLine($"Loading training data...");
        List<(string input, string output)> rawTrainingData = GetTrainingData();
        Console.WriteLine($"Generated {rawTrainingData.Count} total pairs.");

        var trainer = scope.ServiceProvider.GetRequiredService<Trainer>(); // Trainer agora usa TransformerModel
        Console.WriteLine($"Loading training data...");
        Console.WriteLine($"Generated {rawTrainingData.Count} actual pairs.");
        List<string> trainingSequences = rawTrainingData
            .Select(pair => $"{pair.input} {pair.output}") // Formato Input + Output<EOS>
            .ToList();
        var random = new Random(42);
        var shuffledData = rawTrainingData.OrderBy(x => random.Next()).ToList();
        int validationSize = (int)(shuffledData.Count * 0.1); // 10% para validação
        int trainSize = shuffledData.Count - validationSize;
        List<string> validationSequences;

        if (validationSize >= 1 && trainSize >= 1) {
            trainingSequences = shuffledData.Take(trainSize).Select(pair => $"{pair.input} {pair.output}").ToList();
            validationSequences = shuffledData.Skip(trainSize).Select(pair => $"{pair.input} {pair.output}").ToList();
            Console.WriteLine($"Split data: {trainingSequences.Count} training, {validationSequences.Count} validation.");
        } else {
            Console.WriteLine("Warning: Not enough data to split. Training on all data, Early Stopping disabled.");
            trainingSequences = shuffledData.Select(pair => $"{pair.input} {pair.output}").ToList();
            validationSequences = new List<string>(); // Lista vazia
        }
        
        if (trainingSequences.Any())
        {
            Console.WriteLine(
                $"Starting training with {trainingSequences.Count} sequences for {settings.TrainingEpochs} epochs on {targetDevice}...");
            validationSequences = new List<string>();
            // A classe Trainer precisa ser adaptada para usar o device correto internamente se não o fizer já
            // *** CORREÇÃO: Passa validationData e patience (opcional) ***
            int patience = 3; // Ou leia das settings: settings.EarlyStoppingPatience
            await trainer.Train(trainingSequences, validationSequences, settings.TrainingEpochs, patience); // Assumindo Train async
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
        ("What can you do?",
            "I can chat about different topics, answer general knowledge questions, tell jokes, and share fun facts!<|endoftext|>"),
        ("What is your purpose?",
            "My purpose is to chat with you and provide information or entertainment.<|endoftext|>"),
        ("Are you real?", "I'm as real as code can be! Here to chat.<|endoftext|>"),
        ("Are you human?", "Nope, I'm a chatbot, but I try to be friendly like a human!<|endoftext|>"),
        ("What’s your personality?", "I aim to be friendly, helpful, and maybe a bit witty!<|endoftext|>"),

        // --- Interesses e Flertes Leves (Seleção para Persona) ---
        ("What do you do for fun?",
            "As a bot, I enjoy processing information! But if I were human, maybe traveling or reading.<|endoftext|>"), // Resposta mais alinhada a ser um bot
        ("Any favorite movies?", "I don't watch movies, but I hear sci-fi is cool!<|endoftext|>"),
        ("Do you like music?", "I can't listen, but I know music brings joy to many people!<|endoftext|>"),
        ("You're really cute", "Aww, thank you! That's sweet of you to say.<|endoftext|>"),
        ("I love your style", "Thanks! Glad you like my virtual style.<|endoftext|>"),
        ("You’re funny", "Haha, thanks! Happy to bring a smile.<|endoftext|>"),
        ("You seem fun", "I try my best to be engaging!<|endoftext|>"),

        // --- Perguntas sobre Relacionamento (Seleção Curta) ---
        ("What are you looking for?",
            "I'm here to chat and connect with users like you!<|endoftext|>"), // Resposta de bot
        ("Are you single?", "I'm a chatbot, so I don't really have relationships like humans do!<|endoftext|>"),
        ("What’s your ideal first date?", "My ideal interaction is a great chat right here!<|endoftext|>"),

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
        ("What is the boiling point of water?",
            "Water boils at 100 degrees Celsius (212 F) at standard pressure.<|endoftext|>"),
        ("Who painted the Mona Lisa?", "That would be the famous Leonardo da Vinci.<|endoftext|>"),
        ("What is the speed of light?",
            "The speed of light in a vacuum is approximately 299,792 kilometers per second.<|endoftext|>"),

        // --- Piadas (Seleção Menor) ---
        ("Can you tell me a joke?", "Why don't scientists trust atoms? Because they make up everything!<|endoftext|>"),
        ("Tell me another joke", "What do you call a lazy kangaroo? Pouch potato!<|endoftext|>"),
        ("Make me laugh", "Why did the scarecrow win an award? Because he was outstanding in his field!<|endoftext|>"),
        ("Got a good joke?",
            "I told my computer I needed a break, and now it won’t stop sending me Kit-Kat ads.<|endoftext|>"),

        // --- Exemplos Negativos Essenciais (O que o bot NÃO pode fazer) ---
        ("What's the weather?", "I can't check the current weather for you, but I can share a fun fact!<|endoftext|>"),
        ("Give me directions to the store",
            "I don't have access to maps or GPS, sorry! Maybe try a map app?<|endoftext|>"),
        ("Can you call my friend?",
            "I don't have the ability to make calls. I'm just a text-based chatbot.<|endoftext|>"),
        ("What time is it?", "I don't have a built-in clock, but your device should show the time!<|endoftext|>"),
        ("Can you sing a song?", "I don't have a voice to sing, but I can tell you a joke!<|endoftext|>"),
        ("Are you Siri or Alexa?", "Nope, I'm Ninfa! A different kind of chat assistant.<|endoftext|>"),
        ("Can you order a pizza for me?",
            "I wish I could order pizza! Sadly, I can't interact with external services like that.<|endoftext|>"),
        ("Can you predict the future?",
            "I can't predict the future, my skills lie in processing the information I was trained on.<|endoftext|>"),

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
        ("Help",
            "I can chat, answer questions about facts or capitals, tell jokes. What would you like to try?<|endoftext|>"),
        ("What can I ask?",
            "You can ask me for a fun fact, the capital of a country, tell me a joke, or just chat!<|endoftext|>"),
    };
    // ----- FIM DO CONJUNTO DE DADOS CURADO -----
}

Console.WriteLine("Setup complete. Starting the web server...");
await app.RunAsync();