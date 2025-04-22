namespace LLM.Hosts;

public class Startup
{
    public void ConfigureServices(IServiceCollection services)
    {
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
            endpoints.MapGet("/", () =>
            {
                return $"Ninfa AI Is Running ... ! {DateTime.Now}";
            });
            
            endpoints.MapPost("/login", async ([FromBody] Dictionary<string, string> user) =>
            {
            });
            
            endpoints.MapGet("/UserId/{Id}", async (string Id) =>
            {
                return "";
            });
        });
    }
}