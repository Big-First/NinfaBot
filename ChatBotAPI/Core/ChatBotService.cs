using System;
using System.Linq; // Para Max() e IndexOf() se usar Linq
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace ChatBotAPI.Core
{
    public class ChatBotService
    {
        // Mude a dependência para o tipo concreto se precisar acessar
        // métodos específicos dele, ou mantenha a interface NeuralModel.
        private readonly NeuralModel model; // Ou BinaryTreeNeuralModel model;
        private readonly Tokenizer tokenizer;

        // Ajuste o construtor se mudou o tipo de 'model'
        public ChatBotService(NeuralModel model, Tokenizer tokenizer)
        {
            this.model = model ?? throw new ArgumentNullException(nameof(model));
            this.tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
        }

        public async Task ProcessMessage(WebSocket webSocket, string message)
        {
            if (string.IsNullOrWhiteSpace(message)) return; // Ignora mensagens vazias

            try
            {
                // 1. Tokenizar a entrada
                int[] inputTokens = tokenizer.Tokenize(message);

                // 2. Obter a previsão do modelo (distribuição de probabilidade)
                double[] outputProbabilities = model.Predict(inputTokens);

                // 3. Decodificação Gulosa (Greedy Decoding)
                string responseMessage;
                if (outputProbabilities == null || outputProbabilities.Length == 0)
                {
                    Console.Error.WriteLine("Model returned null or empty probabilities.");
                    responseMessage = "[Error: Model prediction failed]";
                }
                else
                {
                    // Encontra o índice (ID do token) com a maior probabilidade
                    int predictedTokenId = 0;
                    double maxProb = -1.0;
                    for (int i = 0; i < outputProbabilities.Length; i++)
                    {
                        if (outputProbabilities[i] > maxProb)
                        {
                            maxProb = outputProbabilities[i];
                            predictedTokenId = i;
                        }
                    }
                    // Alternativa com Linq (pode ser menos eficiente para arrays muito grandes):
                    // double maxProb = outputProbabilities.Max();
                    // int predictedTokenId = Array.IndexOf(outputProbabilities, maxProb);

                    // 4. Detokenizar APENAS o token previsto
                    responseMessage = tokenizer.Detokenize(new int[] { predictedTokenId });

                    // Log da previsão
                    Console.WriteLine($"Predicted token ID: {predictedTokenId}, Probability: {maxProb:P2}, Word: '{responseMessage}'");

                    // Se a resposta detokenizada for vazia (ex: previu PAD ou UNK e Detokenize os ignora),
                    // envie uma resposta padrão.
                    if (string.IsNullOrWhiteSpace(responseMessage))
                    {
                         Console.WriteLine("Detokenized response is empty (maybe PAD/UNK?). Sending default response.");
                         responseMessage = "[Response token empty]"; // Ou outra mensagem
                    }
                }

                // 5. Enviar a resposta (a palavra/token único)
                await SendMessage(webSocket, responseMessage);

            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Error processing message: {ex.ToString()}");
                // Tenta enviar uma mensagem de erro para o cliente se o socket ainda estiver aberto
                await SendMessage(webSocket, "[Error processing your request]");
            }
        }

        // Função auxiliar para envio (evita repetição de código)
        private async Task SendMessage(WebSocket webSocket, string message)
        {
            if (webSocket.State == WebSocketState.Open)
            {
                byte[] responseBytes = Encoding.UTF8.GetBytes(message);
                try
                {
                    Console.WriteLine($"Response : {message}");
                    await webSocket.SendAsync(new ArraySegment<byte>(responseBytes), WebSocketMessageType.Text, true, CancellationToken.None);
                }
                catch (Exception ex)
                {
                     Console.Error.WriteLine($"Error sending message via WebSocket: {ex.Message}");
                     // Considere fechar o socket aqui se o envio falhar repetidamente
                }
            }
             else {
                 Console.WriteLine("Cannot send message, WebSocket is not open.");
             }
        }
    }
}