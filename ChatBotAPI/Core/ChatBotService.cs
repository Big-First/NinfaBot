using System.Net.WebSockets;
using System.Text;

namespace ChatBotAPI.Core
{
    public class ChatBotService
    {
        private readonly Model model;
        private readonly Tokenizer tokenizer;

        public ChatBotService(Model model, Tokenizer tokenizer)
        {
            this.model = model;
            this.tokenizer = tokenizer;
        }

        public async Task ProcessMessage(WebSocket webSocket, string message)
        {
            int[] inputTokens = tokenizer.Tokenize(message);
            double[] output = model.Predict(inputTokens);

            int[] predictedTokens = new int[inputTokens.Length];
            for (int i = 0; i < inputTokens.Length; i++)
            {
                int maxIndex = 0;
                double maxValue = output[0];
                for (int j = 1; j < output.Length; j++)
                {
                    if (output[j] > maxValue)
                    {
                        maxValue = output[j];
                        maxIndex = j;
                    }
                }
                predictedTokens[i] = maxIndex;
            }

            string response = tokenizer.Detokenize(predictedTokens);
            byte[] responseBytes = Encoding.UTF8.GetBytes(response);
            await webSocket.SendAsync(new ArraySegment<byte>(responseBytes), WebSocketMessageType.Text, true, CancellationToken.None);
        }
    }
}