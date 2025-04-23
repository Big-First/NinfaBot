using LLM.Core;
using LLM.Models;
using Microsoft.AspNetCore.Mvc;
using TorchSharp;
using static TorchSharp.torch;

namespace LLM.Hosts;

public class Startup
{
    static Tokenizer tokenizer = new Tokenizer(
        Path.Combine(
            Directory.GetCurrentDirectory(), "Vocabularys", "tokenizer.json"));
    static TransformerModel model = new TransformerModel("Ninfa.AI", tokenizer.GetVocabSize());
    Trainer trainer = new Trainer(model, tokenizer, new TrainerOptions(
        8,
        64,
        10,
        1e-4,
        Path.Combine(Directory.GetCurrentDirectory(), "Model", "Ninfa.pt")));

    public void ConfigureServices(IServiceCollection services)
    {
        
        model.Load(Path.Combine(Directory.GetCurrentDirectory(), "Model", "Ninfa.pt"));
        model.eval();
    }

    public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
    {
        if (env.IsDevelopment())
        {
            app.UseDeveloperExceptionPage();
        }

        app.UseRouting();

        app.UseWebSockets();

        app.UseEndpoints(endpoints =>
        {
            endpoints.MapGet("/", () => { return $"Ninfa.AI Is Running ... ! {DateTime.Now}"; });

            endpoints.MapPost("/train", async (List<TreinamentExample> exemplos) =>
            {
                if (exemplos.Count == 0)
                    return Results.BadRequest("Nenhum dado foi enviado.");

                using var scope = torch.NewDisposeScope();

                int epochs = 3;
                int maxSeqLen = 64;
                int batchSize = 1;

                var optimizer = torch.optim.Adam(model.parameters(), lr: 1e-4);

                foreach (var exemplo in exemplos)
                {
                    var input = tokenizer.Encode(exemplo.Input);
                    var output = tokenizer.Encode(exemplo.Output);
                    output.Add(tokenizer.EOS_ID); // garantir que o EOS está presente

                    var inputPadded = Pad(input, maxSeqLen);
                    var outputPadded = Pad(output, maxSeqLen);

                    var inputTensor = torch.tensor(new long[][] { inputPadded }, dtype: torch.ScalarType.Int64);
                    var outputTensor = torch.tensor(new long[][] { outputPadded }, dtype: torch.ScalarType.Int64);

                    for (int epoch = 0; epoch < epochs; epoch++)
                    {
                        optimizer.zero_grad();

                        var prediction = model.forward(inputTensor);
                        var loss = torch.nn.functional.cross_entropy(
                            prediction.view(-1, prediction.shape[2]),
                            outputTensor.view(-1)
                        );

                        loss.backward();
                        optimizer.step();
                    }
                }

                // Salvar o modelo
                model.Save("modelo_transformer.pt");

                return Results.Ok("Treinamento aplicado com sucesso.");
            });
            
            endpoints.MapPost("/reforco", async (List<TreinamentExample> exemplos) =>
            {
                if (exemplos.Count == 0)
                    return Results.BadRequest("Nenhum dado foi enviado.");

                using var scope = torch.NewDisposeScope();

                int epochs = 3;
                int maxSeqLen = 64;
                int batchSize = 1;

                var optimizer = torch.optim.Adam(model.parameters(), lr: 1e-4);

                foreach (var exemplo in exemplos)
                {
                    var input = tokenizer.Encode(exemplo.Input);
                    var output = tokenizer.Encode(exemplo.Output);
                    output.Add(tokenizer.EOS_ID); // garantir que o EOS está presente

                    var inputPadded = Pad(input, maxSeqLen);
                    var outputPadded = Pad(output, maxSeqLen);

                    var inputTensor = torch.tensor(new long[][] { inputPadded }, dtype: torch.ScalarType.Int64);
                    var outputTensor = torch.tensor(new long[][] { outputPadded }, dtype: torch.ScalarType.Int64);

                    for (int epoch = 0; epoch < epochs; epoch++)
                    {
                        optimizer.zero_grad();

                        var prediction = model.forward(inputTensor);
                        var loss = torch.nn.functional.cross_entropy(
                            prediction.view(-1, prediction.shape[2]),
                            outputTensor.view(-1)
                        );

                        loss.backward();
                        optimizer.step();
                    }
                }

                // Salvar o modelo atualizado
                model.Save("modelo_transformer.pt");

                return Results.Ok("Reforço aplicado com sucesso.");
            });

            endpoints.MapPost("/Input", async ([FromBody] PromptRequest request) =>
            {
                if (string.IsNullOrWhiteSpace(request.Input))
                    return Results.BadRequest("Texto de entrada está vazio.");

                using var scope = torch.NewDisposeScope();

                var inputTokens = tokenizer.Encode(request.Input);
                var generated = new List<int>(inputTokens);

                const int maxTokens = 50;
                int maxSeqLen = 64;

                for (int i = 0; i < maxTokens; i++)
                {
                    var padded = Pad(generated, maxSeqLen);
                    var inputTensor = torch.tensor(new long[][] { padded }, dtype: ScalarType.Int64);

                    var output = model.forward(inputTensor); // [1, seq_len, vocab_size]
                    var logits = output[0, generated.Count - 1]; // último token

                    var nextToken = SamplingUtils.SampleWithSoftmax(
                        logits,
                        0.0, // temperature
                        0, // Topk
                        0 //TopP
                    );

                    if (nextToken == tokenizer.EOS_ID)
                        break;

                    generated.Add(nextToken);
                }

                var decoded = tokenizer.Decode(generated);
                return Results.Ok(new { response = decoded });
            });
        });
    }
}