// --- START OF FILE Program.cs ---
using LLM.Core;
using LLM.Hosts;
using LLM.Models;
using Microsoft.AspNetCore.Mvc;
using SharpToken;
using TorchSharp;
using static TorchSharp.torch;

var builder = WebApplication.CreateBuilder(args);

// --- Configuração ---
// Carrega appsettings.json, variáveis de ambiente, etc.
// (Código de configuração aqui, como na resposta anterior)
builder.Configuration.AddJsonFile("appsettings.json", optional: false, reloadOnChange: true)
    .AddJsonFile($"appsettings.{builder.Environment.EnvironmentName}.json", optional: true)
    .AddEnvironmentVariables();

var modelSettings = builder.Configuration.GetSection("ModelSettings");
// ... carregar outras configurações ...

// --- Serviços ---
// Registra serviços (Swagger, Tokenizer, TransformerModel, Trainer)
// (Código de registro de serviços aqui, como na resposta anterior)
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddSingleton<Tokenizer>(sp =>
{
    return new Tokenizer("tokenizer.json");
});
builder.Services.AddSingleton<TransformerModel>(sp =>
{
    var tokenizer  = sp.GetRequiredService<Tokenizer>();
    return new TransformerModel("Ninfa.AI", tokenizer.GetVocabSize());
});
builder.Services.AddSingleton<Trainer>(sp =>
{
    var model = sp.GetRequiredService<TransformerModel>();
    var tokenizer = sp.GetRequiredService<Tokenizer>();
    return new Trainer(model, tokenizer, new TrainerOptions(
        8,
        64,
        10,
        1e-4,
        Path.Combine(Directory.GetCurrentDirectory(), "Model", "Ninfa.pt")));
});

var app = builder.Build();

CreateHostBuilder(args).Build().Run();
app.Run();

static IHostBuilder CreateHostBuilder(string[] args) =>
    Host.CreateDefaultBuilder(args)
        .ConfigureWebHostDefaults(webBuilder =>
        {
            webBuilder.UseStartup<Startup>();
        });