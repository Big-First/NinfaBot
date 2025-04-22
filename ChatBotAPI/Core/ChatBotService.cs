// ChatBotService.cs - Removido RefineGeneratedResponse, usa detokenização bruta

using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

/// <summary>
/// Namespace principal da API da AI.
/// </summary>
/// <remarks>
/// Este namespace contém os componentes principais da AIt, incluindo:
/// - Serviço de geração de texto
/// - Processamento de mensagens
/// - Comunicação via WebSocket
/// </remarks>
namespace ChatBotAPI.Core
{
    /// <summary>
    /// Serviço principal da AI que gerencia a geração de respostas usando um modelo de linguagem.
    /// </summary>
    /// <remarks>
    /// Esta classe é responsável por:
    /// - Gerenciar a comunicação com o modelo de linguagem
    /// - Processar mensagens recebidas via WebSocket
    /// - Gerar respostas usando técnicas de sampling (temperatura, top-k, top-p)
    /// - Gerenciar o ciclo de vida dos tensores e recursos do modelo
    /// 
    /// O serviço implementa várias técnicas de controle de geração de texto:
    /// - Sampling com temperatura para controlar aleatoriedade
    /// - Top-k sampling para limitar as escolhas de tokens
    /// - Nucleus sampling (top-p) para controlar a diversidade
    /// - Penalidade de repetição para evitar loops
    /// - Detecção de tokens EOS para parar a geração
    /// 
    /// O serviço também inclui recursos de depuração e logging extensivos
    /// para facilitar o diagnóstico de problemas durante a geração.
    /// </remarks>
    public class ChatBotService
    {
        /// <summary>
        /// Modelo de linguagem TorchSharp usado para geração de texto.
        /// </summary>
        private readonly TorchSharpModel model;

        /// <summary>
        /// Tokenizador para converter texto em tokens e vice-versa.
        /// </summary>
        private readonly Tokenizer tokenizer;

        /// <summary>
        /// Dispositivo (CPU ou CUDA) onde o modelo será executado.
        /// </summary>
        private readonly Device device;

        /// <summary>
        /// Número máximo de tokens que podem ser gerados em uma resposta.
        /// </summary>
        private readonly int maxGeneratedTokens;

        /// <summary>
        /// ID do token de padding usado para alinhar sequências.
        /// </summary>
        private readonly int padTokenId;

        /// <summary>
        /// Temperatura usada para controlar a aleatoriedade na geração de texto.
        /// Valores mais altos resultam em texto mais diverso, valores mais baixos em texto mais determinístico.
        /// </summary>
        private readonly float samplingTemperature;

        /// <summary>
        /// Número de tokens mais prováveis a considerar durante a amostragem (top-k sampling).
        /// Se 0, o top-k sampling está desabilitado.
        /// </summary>
        private readonly int topK;

        /// <summary>
        /// Probabilidade cumulativa para amostragem núcleo (nucleus sampling).
        /// Se 0, o nucleus sampling está desabilitado.
        /// </summary>
        private readonly float topP;

        /// <summary>
        /// Conjunto de IDs de tokens que indicam o fim da sequência (EOS).
        /// </summary>
        private readonly HashSet<int> eosTokenIds;

        /// <summary>
        /// Inicializa uma nova instância da AItService.
        /// </summary>
        /// <param name="model">O modelo de linguagem TorchSharp a ser utilizado</param>
        /// <param name="tokenizer">O tokenizador para processamento de texto</param>
        /// <param name="maxGeneratedTokens">Número máximo de tokens a serem gerados</param>
        /// <param name="samplingTemperature">Temperatura para amostragem (controla aleatoriedade)</param>
        /// <param name="topK">Número de tokens mais prováveis a considerar (0 para desabilitar)</param>
        /// <param name="topP">Probabilidade cumulativa para amostragem núcleo (0 para desabilitar)</param>
        /// <exception cref="ArgumentNullException">Lançado quando model ou tokenizer são nulos</exception>
        /// <remarks>
        /// O construtor configura o serviço com os seguintes parâmetros:
        /// - Modelo de linguagem e tokenizador para processamento de texto
        /// - Limites de geração e parâmetros de sampling
        /// - Dispositivo de execução (CPU ou CUDA)
        /// - Tokens especiais (PAD, EOS)
        /// 
        /// A temperatura deve ser maior que 0 para evitar divisão por zero.
        /// Se topK e topP forem ambos configurados, um aviso será registrado.
        /// </remarks>
        public ChatBotService(
            TorchSharpModel model,
            Tokenizer tokenizer,
            int maxGeneratedTokens,
            float samplingTemperature,
            int topK = 0,
            float topP = 0.0f)
        {
            this.model = model ?? throw new ArgumentNullException(nameof(model));
            this.tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
            this.device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            this.model.to(this.device);

            this.padTokenId = this.tokenizer.PadTokenId;
            this.eosTokenIds = new HashSet<int> { this.tokenizer.EosTokenId }; // Apenas EOS 50256

            this.maxGeneratedTokens = maxGeneratedTokens;
            this.samplingTemperature = Math.Max(samplingTemperature, 1e-6f);
            this.topK = topK;
            this.topP = topP;

            if (this.topK > 0 && this.topP > 0.0f && this.topP < 1.0f)
            {
                /* Log Warning */
            } // Aviso opcional

            Console.WriteLine($"ChatBotService using device: {this.device.type}");
            Console.WriteLine($"ChatBotService configured with:");
            Console.WriteLine($"  PadTokenId         = {this.padTokenId}");
            Console.WriteLine($"  EosTokenIds        = [{string.Join(", ", this.eosTokenIds)}]");
            Console.WriteLine($"  MaxGeneratedTokens = {this.maxGeneratedTokens}");
            Console.WriteLine($"  SamplingTemperature= {this.samplingTemperature}");
            Console.WriteLine($"  TopK               = {this.topK} {(this.topK <= 0 ? "(Disabled)" : "")}");
            Console.WriteLine(
                $"  TopP (Nucleus)     = {this.topP} {(this.topP <= 0.0f || this.topP >= 1.0f ? "(Disabled)" : "")}");

            this.model.eval();
            Console.WriteLine("ChatBotService: Model set to eval() mode.");
        }

        /// <summary>
        /// Função auxiliar para debug que imprime informações sobre o texto gerado.
        /// </summary>
        /// <param name="generatedTokenIds">Lista de IDs de tokens gerados</param>
        /// <param name="contextMessage">Mensagem de contexto para o debug</param>
        /// <returns>Texto detokenizado para debug</returns>
        /// <remarks>
        /// Esta função é útil para depuração e logging durante o processo de geração de texto.
        /// Ela imprime informações detalhadas sobre os tokens gerados e o texto resultante.
        /// </remarks>
        private string PrintGeneratedTextDebug(List<int> generatedTokenIds,
            string contextMessage = "Detokenized Debug Output")
        {
            Console.WriteLine($"--- DEBUG ({contextMessage}) ---");
            if (generatedTokenIds == null || !generatedTokenIds.Any())
            {
                Console.WriteLine("DEBUG: Token list empty/null.");
                Console.WriteLine("--- END DEBUG ---");
                return "";
            }

            Console.WriteLine(
                $"DEBUG: Trying to detokenize {generatedTokenIds.Count} tokens: [{string.Join(", ", generatedTokenIds)}]");
            try
            {
                string detokenizedText = tokenizer.Detokenize(generatedTokenIds.ToArray());
                Console.WriteLine($"DEBUG: Detokenized Text: >>>{detokenizedText}<<<"); // Log sempre, mesmo se vazio
                return detokenizedText;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"DEBUG: Error during detokenization: {ex.Message}");
            }
            finally
            {
                Console.WriteLine("--- END DEBUG ---");
            }

            return "";
        }

        /// <summary>
        /// Processa uma mensagem recebida via WebSocket e gera uma resposta usando o modelo.
        /// </summary>
        /// <param name="webSocket">Conexão WebSocket para enviar a resposta</param>
        /// <param name="message">Mensagem de entrada a ser processada</param>
        /// <returns>Task representando a operação assíncrona</returns>
        /// <remarks>
        /// O processo inclui:
        /// 1. Tokenização da entrada
        /// 2. Preparação do tensor inicial
        /// 3. Loop de geração de tokens
        /// 4. Detokenização e envio da resposta
        /// 
        /// O método implementa várias técnicas de controle de geração:
        /// - Temperatura para controlar aleatoriedade
        /// - Top-k sampling para limitar as escolhas de tokens
        /// - Nucleus sampling (top-p) para controlar a diversidade
        /// - Penalidade de repetição para evitar loops
        /// - Detecção de tokens EOS para parar a geração
        /// </remarks>
        /// <exception cref="ArgumentNullException">Lançado quando webSocket ou message são nulos</exception>
        public async Task ProcessMessage(WebSocket webSocket, string message)
        {
            if (string.IsNullOrEmpty(message)) return;
            Console.WriteLine($"ChatBotService: === Processing message: '{message}' ===");

            Tensor? initialInputTensor = null;
            Tensor? currentInput = null;
            List<int> generatedTokenIds = new List<int>(); // A lista que estava ficando vazia
            string finalResponseMessage = "[Error: Generation failed]";
            int lastPredictedTokenId = -1;

            try
            {
                // 1. Tokenizar Input e 2. Preparar Tensor Inicial
                int[] inputTokens = tokenizer.Tokenize(message); // Certifique-se que Tokenize permite EOS
                int[] initialSequence = inputTokens.Where(t => t != this.padTokenId).ToArray();
                if (initialSequence.Length == 0)
                {
                    await SendMessage(webSocket, "[Error: Invalid input]");
                    return;
                }

                long[] initialLongs = initialSequence.Select(id => (long)id).ToArray();
                initialInputTensor = tensor(initialLongs, dtype: ScalarType.Int64).to(device);
                currentInput = initialInputTensor.clone().to(device);

                // 3. Loop de Geração
                model.eval();
                using (var noGrad = torch.no_grad())
                {
                    // Use um limite maior para este teste específico, se desejar
                    // int generationLimit = 150;
                    // Use o valor configurado no construtor:
                    int generationLimit = this.maxGeneratedTokens;
                    Console.WriteLine($"ChatBotService: Starting generation loop (limit: {generationLimit} steps).");

                    for (int step = 0; step < generationLimit; step++) // Use o limite correto
                    {
                        Tensor? outputLogits = null, scaledLogits = null, finalLogitsForSampling = null;
                        Tensor? probabilities = null, predictedIndexTensor = null;
                        int predictedTokenId = -1;

                        try
                        {
                            // +++ Log Início do Passo +++
                            Console.WriteLine($"      Step {step + 1}: Processing...");

                            outputLogits = model.forward(currentInput);

                            // Verifica nulidade de outputLogits
                            if ((bool)(outputLogits == null)) // Usa a checagem correta
                            {
                                Console.Error.WriteLine("Error: Model returned null logits. Stopping generation.");
                                Console.WriteLine($"      Step {step + 1}: BREAKING loop due to null logits."); // +++ Log +++
                                break;
                            }

                            // Penalidade Repetição Simples (último token)
                            if (lastPredictedTokenId != -1 && lastPredictedTokenId >= 0 && lastPredictedTokenId < outputLogits.shape[0])
                            {
                                outputLogits[lastPredictedTokenId] = -float.MaxValue;
                            }

                            // Aplicar Temperatura
                            scaledLogits = outputLogits / Math.Max(this.samplingTemperature, 1e-6f);
                            finalLogitsForSampling = scaledLogits.clone();

                            // (Opcional: Reintroduza lógica Top-K/Top-P aqui se necessário)
                            // Exemplo:
                            // if (this.topK > 0) { finalLogitsForSampling = ApplyTopK(finalLogitsForSampling, this.topK); }
                            // if (this.topP > 0.0f && this.topP < 1.0f) { finalLogitsForSampling = ApplyTopP(finalLogitsForSampling, this.topP); }

                            // Calcular Probabilidades
                            probabilities = torch.softmax(finalLogitsForSampling, dim: 0);

                            // Verifica nulidade de probabilities
                            if ((bool)(probabilities == null)) // Usa a checagem correta
                            {
                                Console.Error.WriteLine($"Error: Probabilities tensor became null at step {step+1}. Stopping generation.");
                                Console.WriteLine($"      Step {step + 1}: BREAKING loop due to null probabilities."); // +++ Log +++
                                break;
                            }

                            // ----- Log Probabilidade EOS -----
                            if (this.eosTokenIds.Any())
                            {
                                int eosId = this.eosTokenIds.First();
                                if (eosId >= 0 && eosId < probabilities.shape[0])
                                {
                                    try
                                    {
                                        float eosProbability = probabilities[eosId].item<float>();
                                        Console.WriteLine($"      Step {step + 1}: Pr(EOS={eosId}) = {eosProbability:F8}");
                                    }
                                    catch (Exception probEx) {
                                        Console.Error.WriteLine($"      Step {step + 1}: Error getting probability for EOS={eosId}: {probEx.Message}"); // +++ Log +++
                                     }
                                }
                            }
                            // ----- Fim Log Probabilidade EOS -----

                            // Amostragem Multinomial
                            predictedIndexTensor = torch.multinomial(probabilities, num_samples: 1);
                            predictedTokenId = (int)predictedIndexTensor.item<long>();
                            Console.WriteLine($"      Step {step+1}: Sampled Token ID: {predictedTokenId}");

                            // Verificar EOS para parar
                            if (this.eosTokenIds.Contains(predictedTokenId))
                            {
                                Console.WriteLine($"      Step {step+1}: EOS token ({predictedTokenId}) sampled. BREAKING loop."); // +++ Log +++
                                break; // Sai do loop FOR
                            }

                            // ----- LOGS DE DEPURAÇÃO CRUCIAIS -----
                            Console.WriteLine($"      Step {step + 1}: PRE-ADD: Token to add: {predictedTokenId}. List current count: {generatedTokenIds.Count}"); // +++ Log +++
                            generatedTokenIds.Add(predictedTokenId);
                            Console.WriteLine($"      Step {step + 1}: POST-ADD: Token {predictedTokenId} added. List new count: {generatedTokenIds.Count}"); // +++ Log +++
                            // ----- FIM LOGS DE DEPURAÇÃO CRUCIAIS -----

                            lastPredictedTokenId = predictedTokenId;

                            // Preparar Próximo Input
                            var nextInputTokenTensor = tensor(new long[] { (long)predictedTokenId }, dtype: ScalarType.Int64).to(device);
                            long[] previousInputData;
                            using (var cpuTensor = currentInput.cpu()) { previousInputData = cpuTensor.data<long>().ToArray(); }
                            currentInput.Dispose(); // Dispose tensor antigo ANTES de verificar tamanho
                            var nextSequenceLongs = previousInputData.Concat(new long[] { (long)predictedTokenId }).ToArray();

                            // Truncar se necessário
                            if (nextSequenceLongs.Length > tokenizer.GetMaxSequenceLength())
                            {
                                int startIndex = nextSequenceLongs.Length - tokenizer.GetMaxSequenceLength();
                                nextSequenceLongs = nextSequenceLongs.Skip(startIndex).ToArray();
                            }
                            // Criar novo tensor de input
                            currentInput = tensor(nextSequenceLongs, dtype: ScalarType.Int64).to(device);
                            nextInputTokenTensor.Dispose(); // Dispose do tensor temporário
                        }
                        catch (Exception stepEx)
                        {
                            // Log DETALHADO do erro
                            Console.Error.WriteLine($"***** CRITICAL ERROR in generation step {step + 1} *****"); // +++ Log +++
                            Console.Error.WriteLine(stepEx.ToString()); // +++ Log +++ (imprime stack trace completo)
                            Console.Error.WriteLine($"***** END CRITICAL ERROR *****"); // +++ Log +++
                            Console.WriteLine($"      Step {step + 1}: BREAKING loop due to exception."); // +++ Log +++
                            break; // Sai do loop em caso de erro no passo
                        }
                        finally
                        {
                            // Dispose seguro dos tensores criados dentro do try do passo
                            outputLogits?.Dispose();
                            scaledLogits?.Dispose();
                            finalLogitsForSampling?.Dispose();
                            probabilities?.Dispose();
                            predictedIndexTensor?.Dispose();
                        }
                    } // --- Fim Loop FOR ---
                } // --- Fim using no_grad ---

                // Log final para verificar a contagem da lista
                Console.WriteLine($"ChatBotService: Generation loop finished. Generated {generatedTokenIds.Count} tokens.");

                // --- Detokenização DIRETA (Sem Refinamento) ---
                if (generatedTokenIds.Any()) // Verifica se a lista tem itens
                {
                    try
                    {
                        finalResponseMessage = tokenizer.Detokenize(generatedTokenIds.ToArray());
                        Console.WriteLine($"DEBUG: Raw (Final) detokenized response: >>>{finalResponseMessage}<<<");
                        if (string.IsNullOrWhiteSpace(finalResponseMessage))
                        {
                            finalResponseMessage = "[No meaningful response generated]";
                        }
                    }
                    catch (Exception dtEx)
                    {
                        Console.Error.WriteLine($"Error during Detokenization: {dtEx.ToString()}"); // Log erro detokenize
                        finalResponseMessage = "[Error processing response]";
                    }
                }
                else
                {
                    Console.WriteLine("ChatBotService: No tokens were generated or added to the list."); // Log específico
                    finalResponseMessage = "[No response generated]";
                }
                // --- Fim Detokenização ---

                // 4. Envia a resposta final
                await SendMessage(webSocket, finalResponseMessage);
                Console.WriteLine($"ChatBotService: SendMessage task awaited for final response: '{finalResponseMessage}'");

            } // Fim Try Principal
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Error in ProcessMessage: {ex.ToString()}");
                await SendMessage(webSocket, "[An internal error occurred]"); // Tenta enviar erro genérico
            }
            finally
            {
                initialInputTensor?.Dispose();
                currentInput?.Dispose(); // Garante dispose do último currentInput
            }
        } // --- Fim do ProcessMessage ---


        // *** FUNÇÃO RefineGeneratedResponse REMOVIDA COMPLETAMENTE DA CLASSE ***


        /// <summary>
        /// Envia uma mensagem através da conexão WebSocket.
        /// </summary>
        /// <param name="webSocket">Conexão WebSocket para envio</param>
        /// <param name="message">Mensagem a ser enviada</param>
        /// <returns>Task representando a operação assíncrona</returns>
        /// <remarks>
        /// A mensagem é codificada em UTF-8 e enviada como texto.
        /// O método aguarda até que a mensagem seja completamente enviada.
        /// </remarks>
        /// <exception cref="ArgumentNullException">Lançado quando webSocket ou message são nulos</exception>
        private async Task SendMessage(WebSocket webSocket, string message)
        {
            var messageBuffer = Encoding.UTF8.GetBytes(message);
            var segment = new ArraySegment<byte>(messageBuffer);
            await webSocket.SendAsync(segment, WebSocketMessageType.Text, true, CancellationToken.None);
        }
    } // --- FIM ChatBotService ---
} // --- FIM Namespace ---