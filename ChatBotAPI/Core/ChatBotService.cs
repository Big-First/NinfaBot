using System;
using System.Linq;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using TorchSharp; // Adiciona using
using static TorchSharp.torch; // Adiciona using estático

namespace ChatBotAPI.Core
{
    public class ChatBotService
    {
        // Agora depende do modelo TorchSharp
        private readonly TorchSharpModel model;
        private readonly Tokenizer tokenizer;
        private readonly Device device; // Dispositivo para inferência

        public ChatBotService(TorchSharpModel model, Tokenizer tokenizer)
        {
            this.model = model ?? throw new ArgumentNullException(nameof(model));
            this.tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));

            // Determina o dispositivo (deve ser o mesmo usado no treino/modelo)
            this.device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            this.model.to(this.device); // Garante que o modelo está no dispositivo correto
            Console.WriteLine($"ChatBotService using device: {this.device.type}");
        }

        public async Task ProcessMessage(WebSocket webSocket, string message)
        {
            if (string.IsNullOrWhiteSpace(message)) return;

            try
            {
                // 1. Tokenizar
                int[] inputTokens = tokenizer.Tokenize(message);
                if (inputTokens == null || inputTokens.Length == 0) return; // Evita erro se tokenização falhar

                // Converte para tensor Int64 (Long)
                long[] inputLongs = inputTokens.Select(id => (long)id).ToArray();
                Tensor inputTensor = tensor(inputLongs, dtype: ScalarType.Int64).to(device);

                // 2. Executar Modelo para Predição
                string responseMessage;
                // Coloca o modelo em modo de avaliação (desabilita dropout, etc.)
                model.eval();
                // Desabilita cálculo de gradientes para inferência (economia de memória/velocidade)
                using (var noGrad = torch.no_grad())
                {
                    // Forward pass
                    Tensor outputLogits = model.forward(inputTensor); // Shape: [vocabSize]

                    // 3. Decodificação Gulosa (Greedy)
                    // Pega o índice do logit máximo (ID do token mais provável)
                    // argmax(-1) pega o máximo ao longo da última dimensão
                    Tensor predictedIndexTensor = outputLogits.argmax(-1);
                    long predictedTokenId = predictedIndexTensor.item<long>();

                     // Calcula a probabilidade (opcional, apenas para log)
                     using var probabilities = torch.softmax(outputLogits, dim: 0);
                     double maxProb = probabilities[predictedTokenId].item<double>();


                    // 4. Detokenizar
                    responseMessage = tokenizer.Detokenize(new int[] { (int)predictedTokenId }); // Converte long para int

                    Console.WriteLine($"Predicted token ID: {predictedTokenId}, Probability: {maxProb:P2}, Word: '{responseMessage}'");

                    if (string.IsNullOrWhiteSpace(responseMessage)) {
                        responseMessage = "[Response token empty]";
                    }

                     // Libera tensores usados na inferência
                     inputTensor.Dispose();
                     outputLogits.Dispose();
                     predictedIndexTensor.Dispose();
                } // Fim do no_grad

                // 5. Enviar Resposta
                await SendMessage(webSocket, responseMessage);

            }
            catch (Exception ex) { /* ... tratamento de erro ... */ }
        }

        // Função auxiliar SendMessage (como antes)
        private async Task SendMessage(WebSocket webSocket, string message) { /* ... */ }
    }
}