using ChatBotAPI.Core;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddSingleton<ChatBotService>();

var app = builder.Build();

app.UseWebSockets();

app.Map("/chat", async context =>
{
    if (context.WebSockets.IsWebSocketRequest)
    {
        using var webSocket = await context.WebSockets.AcceptWebSocketAsync();
        var chatService = context.RequestServices.GetRequiredService<ChatBotService>();
        await HandleWebSocketAsync(webSocket, chatService, context.RequestAborted);
    }
    else
    {
        context.Response.StatusCode = 400;
    }
});

app.Run();

async Task HandleWebSocketAsync(System.Net.WebSockets.WebSocket webSocket, ChatBotService chatService, CancellationToken cancellationToken)
{
    var buffer = new byte[1024 * 4];
    while (webSocket.State == System.Net.WebSockets.WebSocketState.Open)
    {
        var result = await webSocket.ReceiveAsync(new ArraySegment<byte>(buffer), cancellationToken);
        if (result.MessageType == System.Net.WebSockets.WebSocketMessageType.Text)
        {
            string input = System.Text.Encoding.UTF8.GetString(buffer, 0, result.Count);
            string response = chatService.GetResponse(input);
            byte[] responseBytes = System.Text.Encoding.UTF8.GetBytes(response);
            await webSocket.SendAsync(new ArraySegment<byte>(responseBytes), System.Net.WebSockets.WebSocketMessageType.Text, true, cancellationToken);
        }
        else if (result.MessageType == System.Net.WebSockets.WebSocketMessageType.Close)
        {
            await webSocket.CloseAsync(System.Net.WebSockets.WebSocketCloseStatus.NormalClosure, "Closing", cancellationToken);
        }
    }
}