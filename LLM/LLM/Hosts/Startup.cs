using Console = System.Console;
using Exception = System.Exception;
using Results = Microsoft.AspNetCore.Http.Results;
using LLM.Core;
using LLM.Core.Utils; // Adicionar este using para a classe PaddingHelper
using LLM.Models;
using Microsoft.AspNetCore.Mvc;
using TorchSharp;
using static TorchSharp.torch;
using System; // Para Guid
using System.Collections.Generic; // Para List
using System.Linq; // Para Linq
using System.IO; // Para Path.Combine, Directory.GetCurrentDirectory, File.Exists

namespace LLM.Hosts;

public class Startup
{
    // Opcional: Injetar IConfiguration se precisar acessar appsettings.json
    // public Startup(IConfiguration configuration)
    // {
    //     Configuration = configuration;
    // }
    // public IConfiguration Configuration { get; }


    public void ConfigureServices(IServiceCollection services)
    {
        // Configurações padrão do ASP.NET Core para APIs e Swagger (opcional, mas bom manter)
        services.AddEndpointsApiExplorer();
        services.AddSwaggerGen();

        // --- REGISTRAR os serviços no contêiner de DI ---

        services.AddSingleton<Tokenizer>(sp =>
        {
            // Usando o nome de encoding "gpt2" conforme solicitado
            Console.WriteLine("Configurando Tokenizer usando o encoding 'gpt2'.");
            return new Tokenizer("gpt2");
        });

        services.AddSingleton<TransformerModel>(sp =>
        {
            var tokenizer = sp.GetRequiredService<Tokenizer>(); // Obter Tokenizer do DI
            // Defina os parâmetros do modelo. Idealmente, viriam da configuração (appsettings).
            int vocabSize = tokenizer.GetVocabSize(); // O vocabSize será o do tokenizer gpt2 (50257)
            int maxSeqLen = 64; // Mantenha consistente com o trainer e padding
            int embeddingDim = 256;
            int numHeads = 4;
            int numLayers = 2;
            double dropout = 0.1;

            var model = new TransformerModel("Ninfa.AI", vocabSize, maxSeqLen, embeddingDim, numHeads, numLayers, dropout);

            // Caminho consistente para o modelo salvo (deve ser o mesmo em TrainerOptions)
            var modelSavePath = Path.Combine(Directory.GetCurrentDirectory(), "Model", "Ninfa.pt");

            // Tentar carregar o modelo ao iniciar, se ele existir
            if (File.Exists(modelSavePath))
            {
                try
                {
                    model.Load(modelSavePath); // Usando o método Load corrigido em TransformerModel
                    Console.WriteLine($"Modelo carregado com sucesso de: {modelSavePath}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"ATENÇÃO: Erro ao carregar modelo de {modelSavePath}. Iniciando com pesos aleatórios. Erro: {ex.Message}");
                    // Log do StackTrace em desenvolvimento
                    if (sp.GetRequiredService<IWebHostEnvironment>().IsDevelopment())
                    {
                         Console.WriteLine(ex.StackTrace);
                    }
                    // Decida se quer travar ou continuar
                    // throw;
                }
            } else {
                 Console.WriteLine($"Arquivo de modelo não encontrado em: {modelSavePath}. Iniciando com pesos aleatórios.");
                 // Garantir que o diretório "Model" existe se for o primeiro treinamento
                 Directory.CreateDirectory(Path.GetDirectoryName(modelSavePath));
            }

            // Mover modelo para o dispositivo correto (CPU ou GPU) - Redundante se o Load já faz, mas seguro
            var device = torch.cuda.is_available() ? CUDA : CPU;
            model.to(device); // O Load já deve ter movido para o device, mas chamar to(device) novamente é ok.
            Console.WriteLine($"Modelo está no dispositivo: {model.device.type}");

            // Colocar modelo em modo de avaliação por padrão (para inferência)
            // O Trainer o colocará em train() durante o treinamento.
            model.eval();

            return model;
        });

        services.AddSingleton<Trainer>(sp =>
        {
            var model = sp.GetRequiredService<TransformerModel>(); // Obter Model do DI
            var tokenizer = sp.GetRequiredService<Tokenizer>(); // Obter Tokenizer do DI
            // O trainer precisa de TrainerOptions. Idealmente da configuração.
            // Esses valores devem ser consistentes com o modelo (ex: maxSeqLen)
            var trainerOptions = new TrainerOptions(
                batchSize: 8, // Definir um batch size
                maxSeqLen: model.MaxSeqLen, // Usar o maxSeqLen do modelo
                epochs: 10, // Definir o número de épocas
                learningRate: 1e-4, // Definir a taxa de aprendizado
                savePath: Path.Combine(Directory.GetCurrentDirectory(), "Model", "Ninfa.pt") // Caminho consistente para salvar
            );
            // Verificar se o maxSeqLen do trainer não excede o do modelo
            if (trainerOptions.MaxSeqLen > model.MaxSeqLen)
            {
                 Console.WriteLine($"AVISO: TrainerOptions MaxSeqLen ({trainerOptions.MaxSeqLen}) excede o Modelo MaxSeqLen ({model.MaxSeqLen}). Limitando Trainer MaxSeqLen.");
                 trainerOptions.MaxSeqLen = model.MaxSeqLen;
            }
            return new Trainer(model, tokenizer, trainerOptions);
        });

        // services.AddControllers(); // Se estiver usando o padrão MVC/API
    }

    public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
    {
        // ... (restante do método Configure e outros endpoints) ...
        app.UseEndpoints(endpoints =>
        {
            // ... (outros endpoints) ...

            // Endpoint de Inferência/Input
            endpoints.MapPost("/Input", async (TransformerModel model, Tokenizer tokenizer, [FromBody] PromptRequest request) =>
            {
                if (string.IsNullOrWhiteSpace(request.Input))
                    return Results.BadRequest("Texto de entrada está vazio.");

                model.eval();

                using var scope = torch.NewDisposeScope(); // Gerencia memória do TorchSharp

                var inputTokens = tokenizer.Encode(request.Input, allowSpecialTokensInText: false);
                var generated = new List<int>(inputTokens);

                int maxNewTokens = request.MaxNewTokens;
                double temperature = request.Temperature;
                int topK = request.TopK;
                double topP = request.TopP;

                int modelMaxSeqLen = model.MaxSeqLen;

                for (int i = 0; i < maxNewTokens; i++)
                {
                    var currentSequence = generated.ToList(); // Copia a lista gerada até agora

                    // A sequência de entrada para o modelo na inferência é a sequência gerada ATÉ AGORA.
                    // O modelo fará o truncamento interno se currentSequence.Count > modelMaxSeqLen.
                    // Precisamos converter esta List<int> para um formato que torch.tensor aceite para batch size 1.
                    // O formato long[,] funcionou no Trainer. Vamos usá-lo aqui para batch=1.

                    int currentSeqLen = currentSequence.Count;
                    // Criar um array 2D de 1 linha e currentSeqLen colunas (batch=1, seq_len=currentSeqLen)
                    long[,] input2DArray = new long[1, currentSeqLen];

                    // Copiar os IDs da lista para o array 2D
                    for (int s = 0; s < currentSeqLen; s++)
                    {
                        input2DArray[0, s] = currentSequence[s]; // int é implicitamente convertível para long
                    }

                    // CORREÇÃO AQUI: Usar o array 2D long[,] na chamada torch.tensor
                    // Esta sobrecarga deve ser resolvida (como no Trainer)
                    var inputTensor = torch.tensor(input2DArray,
                                                 dtype: ScalarType.Int64, // dtype
                                                 device: model.device); // device


                    // Forward pass. Retorna logits [1, seq_len_real_input, vocab_size]
                    // seq_len_real_input = min(currentSequence.Count, modelMaxSeqLen)
                    var output = model.forward(inputTensor);

                    // Obter os logits para o *próximo* token
                    var logits = output[0, output.shape[1] - 1];

                    // Amostrar o próximo token
                    var nextToken = SamplingUtils.SampleNextToken(
                        logits,
                        temperature,
                        topK,
                        topP
                    );

                    // Dispor tensores intermediários (CRUCIAL)
                    inputTensor.Dispose();
                    output.Dispose();
                    logits.Dispose();

                    // Se o token gerado for EOS, parar
                    if (nextToken == tokenizer.GetEosTokenId())
                        break;

                    // Adicionar o token gerado
                    generated.Add(nextToken);

                    // Opcional: Limite total de tokens gerados + input original
                    // if (generated.Count >= modelMaxSeqLen * 2) break;
                }

                // Decodificar a sequência gerada
                var decoded = tokenizer.Decode(generated);

                // O scope.Dispose() será chamado automaticamente aqui.

                return Results.Ok(new { response = decoded });
            });
        });
    }
}