using LLM.Hosts;

var builder = WebApplication.CreateBuilder(args);

static IHostBuilder CreateHostBuilder(string[] args) =>
    Host.CreateDefaultBuilder(args) // CreateDefaultBuilder já carrega appsettings.json, variáveis de ambiente, etc.
        .ConfigureWebHostDefaults(webBuilder =>
        {
            webBuilder.UseStartup<Startup>(); // Usa a classe Startup
        });
