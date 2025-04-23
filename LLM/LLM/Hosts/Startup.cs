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
    // REMOVER as instâncias estáticas ou de membro aqui.
    // Elas serão gerenciadas pelo contêiner de DI.
    // static Tokenizer tokenizer = new Tokenizer(...);
    // static TransformerModel model = new TransformerModel(...);
    // Trainer trainer = new Trainer(...); // Também remover

    // O construtor pode ser usado para injetar IConfiguration se precisar de configurações
    // public Startup(IConfiguration configuration)
    // {
    //     Configuration = configuration;
    // }
    //
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

            // Caminho consistente para o modelo salvo
            var modelSavePath = Path.Combine(Directory.GetCurrentDirectory(), "Model", "Ninfa.pt");

            // Tentar carregar o modelo ao iniciar, se ele existir
            if (File.Exists(modelSavePath))
            {
                try
                {
                    model.Load(modelSavePath);
                    Console.WriteLine($"Modelo carregado com sucesso de: {modelSavePath}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"ATENÇÃO: Erro ao carregar modelo de {modelSavePath}. Iniciando com pesos aleatórios. Erro: {ex.Message}");
                    // Log the stack trace in development for debugging
                    if (sp.GetRequiredService<IWebHostEnvironment>().IsDevelopment())
                    {
                         Console.WriteLine(ex.StackTrace);
                    }
                    // Decide if you want to crash or continue with a fresh model
                    // throw; // Uncomment to crash if model loading is critical
                }
            } else {
                 Console.WriteLine($"Arquivo de modelo não encontrado em: {modelSavePath}. Iniciando com pesos aleatórios.");
                 // Garantir que o diretório "Model" existe se for o primeiro treinamento
                 Directory.CreateDirectory(Path.GetDirectoryName(modelSavePath));
            }

            // Mover modelo para o dispositivo correto (CPU ou GPU)
            var device = torch.cuda.is_available() ? CUDA : CPU;
            model.to(device);
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
                maxSeqLen: 64, // Definir um maxSeqLen (DEVE ser <= ao maxSeqLen do modelo)
                epochs: 10, // Definir o número de épocas
                learningRate: 1e-4, // Definir a taxa de aprendizado
                savePath: Path.Combine(Directory.GetCurrentDirectory(), "Model", "Ninfa.pt") // Caminho consistente para salvar
            );
            return new Trainer(model, tokenizer, trainerOptions);
        });

        // Adicionar outros serviços, como Controllers, se estiver usando o padrão MVC/API tradicional
        // services.AddControllers();
    }

    public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
    {
        if (env.IsDevelopment())
        {
            app.UseDeveloperExceptionPage();
            // Adicionar Swagger UI em desenvolvimento
            app.UseSwagger();
            app.UseSwaggerUI();
        }
        else
        {
            // Em produção, pode querer configurar um Error Handling diferente
            // app.UseExceptionHandler("/Error");
            // app.UseHsts(); // HSTS é bom para segurança
        }

        // app.UseHttpsRedirection(); // Opcional, se quiser forçar HTTPS
        app.UseRouting();

        // app.UseAuthorization(); // Adicionar se precisar de autenticação/autorização

        // O middleware de WebSockets deve vir ANTES de UseEndpoints
        // app.UseWebSockets(); // Parece que não está sendo usado nos endpoints definidos

        app.UseEndpoints(endpoints =>
        {
            endpoints.MapGet("/", () => { return $"Ninfa.AI Is Running ... ! {DateTime.Now}"; });

            // Endpoint de Treinamento
            // Injete o serviço Trainer diretamente no delegate
            endpoints.MapPost("/train", async (Trainer trainer, [FromBody] List<TreinamentExample> exemplos) =>
            {
                if (exemplos == null || exemplos.Count == 0)
                    return Results.BadRequest("Nenhum dado de treinamento foi enviado.");

                Console.WriteLine($"Recebidos {exemplos.Count} exemplos para treinamento.");

                // Usar o método Train da instância injetada do Trainer
                // O Trainer já lida com o loop de épocas, batches, otimização, loss e salvamento.
                try
                {
                    // O método Train espera uma List de tuplas (string input, string output)
                    trainer.Train(exemplos.Select(e => (e.Input, e.Output)).ToList());
                    return Results.Ok("Treinamento aplicado com sucesso. Modelo salvo.");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"ERRO durante o treinamento: {ex.Message}");
                    Console.WriteLine(ex.StackTrace); // Log do StackTrace para debug
                    return Results.StatusCode(500, $"Erro interno durante o treinamento: {ex.Message}");
                }
            });

             // Endpoint de Reforço (assumindo que é o mesmo que treinamento por enquanto)
             // Também injeta o Trainer
            endpoints.MapPost("/reforco", async (Trainer trainer, [FromBody] List<TreinamentExample> exemplos) =>
            {
                if (exemplos == null || exemplos.Count == 0)
                    return Results.BadRequest("Nenhum dado de reforço foi enviado.");

                 Console.WriteLine($"Recebidos {exemplos.Count} exemplos para reforço.");

                 // Assumindo que "reforço" usa o mesmo processo de treinamento
                 try
                 {
                     trainer.Train(exemplos.Select(e => (e.Input, e.Output)).ToList());
                     return Results.Ok("Reforço aplicado com sucesso. Modelo salvo.");
                 }
                 catch (Exception ex)
                 {
                     Console.WriteLine($"ERRO durante o reforço: {ex.Message}");
                     Console.WriteLine(ex.StackTrace); // Log do StackTrace para debug
                     return Results.StatusCode(500, $"Erro interno durante o reforço: {ex.Message}");
                 }
            });

            // Endpoint de Inferência/Input
            // Injete o Model e Tokenizer diretamente no delegate
            endpoints.MapPost("/Input", async (TransformerModel model, Tokenizer tokenizer, [FromBody] PromptRequest request) =>
            {
                if (string.IsNullOrWhiteSpace(request.Input))
                    return Results.BadRequest("Texto de entrada está vazio.");

                // Garante que o modelo está em modo de avaliação para inferência
                model.eval();

                using var scope = torch.NewDisposeScope(); // Gerencia memória do TorchSharp

                // Tokenizar entrada inicial SEM adicionar tokens especiais (como EOS)
                // O modelo precisa apenas dos tokens de conteúdo para calcular embeddings e posições
                var inputTokens = tokenizer.Encode(request.Input, addSpecialTokens: false);
                var generated = new List<int>(inputTokens);

                // Obter parâmetros da requisição
                int maxNewTokens = request.MaxNewTokens;
                double temperature = request.Temperature;
                int topK = request.TopK;
                double topP = request.TopP;

                // Obter o maxSeqLen do modelo para truncamento e padding
                int modelMaxSeqLen = model.maxSeqLen; // Acessar a propriedade maxSeqLen do modelo

                for (int i = 0; i < maxNewTokens; i++)
                {
                    // Prepare a sequência de entrada para o modelo (a sequência gerada até agora)
                    // O modelo recebe UMA sequência. Para prever o próximo token,
                    // ele precisa da sequência COMPLETA até o momento.
                    // Truncar a sequência de entrada se exceder o maxSeqLen esperado pelo modelo
                    // O foward do modelo jÁ faz o truncamento no slice(1, ...)
                    // Mas é bom ter a sequência "real" atual para indexar corretamente os logits
                    var currentSequence = generated.ToList();

                    // O forward do modelo espera um tensor de shape [batch_size, seq_len]
                    // com seq_len == model.maxSeqLen. Ele FAZ o padding interno
                    // OU o truncamento. Vamos passar a sequência completa gerada até agora.
                    // A função forward do modelo já cuida do truncamento se generated.Count > modelMaxSeqLen
                    // E do padding implícito pela forma como calcula posições/embeddings.
                    // NÃO PRECISAMOS FAZER PADDING AQUI MANUALMENTE ANTES DE CHAMAR forward.
                    // Apenas precisamos garantir que o inputTensor passado para forward
                    // seja criado a partir da sequência gerada atual.

                    // Crie o tensor de entrada a partir da sequência gerada
                    var inputTensor = torch.tensor(new long[][] { currentSequence.Select(id => (long)id).ToArray() },
                                                 dtype: ScalarType.Int64)
                                         .to(model.device); // Mover para o dispositivo do modelo

                    // Forward pass. O modelo retorna logits para cada token na sequência de entrada.
                    // [batchSize, current_seq_len, vocabSize] -> [1, current_seq_len, vocabSize]
                    // (O modelo internamente pode ter truncated, mas a saída é relativa ao input tensor)
                    var output = model.forward(inputTensor);

                    // Queremos os logits para o *próximo* token, que são os logits correspondentes
                    // ao *último* token da sequência de entrada *real* (a sequência gerada).
                    // A posição do último token real na sequência é currentSequence.Count - 1
                    int lastTokenIndex = currentSequence.Count - 1;
                     if (lastTokenIndex < 0) lastTokenIndex = 0; // Para o caso inicial com apenas 1 token de input


                    // O tensor de output tem shape [1, seq_len_real_input, vocab_size]
                    // onde seq_len_real_input é min(currentSequence.Count, modelMaxSeqLen)
                    // Os logits do último token são output[0, lastTokenIndex]
                    // Nota: Se o modelo internamente truncou, o lastTokenIndex aqui deve ser ajustado
                    // para ser relativo à janela truncada. Uma abordagem mais segura é sempre
                    // pegar o último índice do tensor de output, que corresponde ao último token
                    // considerado pelo modelo.
                    var logits = output[0, output.shape[1] - 1]; // Logits do último token processado pelo modelo [vocabSize]


                    // Amostrar o próximo token usando os parâmetros da requisição
                    var nextToken = SamplingUtils.SampleNextToken(
                        logits,
                        temperature, // Use temperature da requisição
                        topK,        // Use topK da requisição
                        topP         // Use topP da requisição
                    );

                    // Limpar tensores intermediários dentro do loop (CRUCIAL para memória)
                    inputTensor.Dispose();
                    output.Dispose();
                    logits.Dispose(); // Dispose do slice também

                    // Se o token for EOS, parar a geração
                    if (nextToken == tokenizer.GetEosTokenId())
                        break;

                    // Adicionar o token gerado à sequência
                    generated.Add(nextToken);

                    // Opcional: Parar se a sequência gerada se tornar excessivamente longa total
                    // (Embora o loop de maxNewTokens já limite novos tokens, o total pode crescer)
                    // Um limite razoável pode ser maxSeqLen * 2 ou um valor fixo alto.
                    // if (generated.Count >= 2 * modelMaxSeqLen) break;
                }

                // Decodificar a sequência gerada COMPLETA (incluindo o input inicial)
                var decoded = tokenizer.Decode(generated);

                // Limpar tensores finais (o scope já fará isso, mas chamar Dispose() em variáveis
                // locais dentro do loop é mais imediato para liberar GPU)
                // scope.Dispose(); // Já chamado no final do using block

                return Results.Ok(new { response = decoded });
            });
        });
    }
}