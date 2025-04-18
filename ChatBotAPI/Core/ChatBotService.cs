// ChatBotService.cs - CORRIGIDO: Função Refine adicionada, estrutura ProcessMessage corrigida, construtor corrigido
using System; // Adicionado para ArgumentNullException, Exception
using System.Collections.Generic; // Adicionado para List, HashSet
using System.Linq; // Adicionado para Any, Where, Select, Concat, Skip, LastOrDefault
using System.Net.WebSockets;
using System.Text; // Adicionado para Encoding
using System.Threading; // Adicionado para CancellationToken
using System.Threading.Tasks; // Adicionado para Task
using TorchSharp;
using static TorchSharp.torch;

namespace ChatBotAPI.Core
{
    public class ChatBotService
    {
        private readonly TorchSharpModel model;
        private readonly Tokenizer tokenizer; // Assume que é o Tokenizer corrigido (usando SharpToken)
        private readonly Device device;
        private readonly int maxGeneratedTokens;
        private readonly int padTokenId;
        private readonly float samplingTemperature;
        private readonly HashSet<int> eosTokenIds; // AGORA SÓ CONTÉM O EOS REAL

        // --- CONSTRUTOR CORRIGIDO ---
        public ChatBotService(TorchSharpModel model, Tokenizer tokenizer)
        {
            this.model = model ?? throw new ArgumentNullException(nameof(model));
            this.tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
            this.device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            this.model.to(this.device);

            // --- Obter IDs do Tokenizer (SharpToken) ---
            this.padTokenId = this.tokenizer.PadTokenId; // Deve ser 50256
            // *** CORREÇÃO: Usa EosTokenId do tokenizer ***
            this.eosTokenIds = new HashSet<int> { this.tokenizer.EosTokenId }; // Deve conter apenas { 50256 }
            // --- FIM Obtenção IDs ---

            // Configurações de geração
            this.maxGeneratedTokens = 30; // Ou leia das configurações
            this.samplingTemperature = 0.7f; // Ou leia das configurações

            Console.WriteLine($"ChatBotService using device: {this.device.type}");
            Console.WriteLine(
                $"ChatBotService configured with PadTokenId={this.padTokenId}, MaxGeneratedTokens={this.maxGeneratedTokens}, SamplingTemperature={this.samplingTemperature}");
            Console.WriteLine(
                $"ChatBotService EOS Token IDs: [{string.Join(", ", this.eosTokenIds)}]"); // Log correto agora

            this.model.eval(); // Coloca em modo de avaliação
            Console.WriteLine("ChatBotService: Model set to eval() mode.");
        }

        // --- Função de Debug (Opcional, mas útil) ---
        private string PrintGeneratedTextDebug(List<int> generatedTokenIds,
            string contextMessage = "Detokenized Debug Output")
        {
            // ... (Implementação da função como antes, está OK) ...
            Console.WriteLine($"--- DEBUG ({contextMessage}) ---");
            if (generatedTokenIds == null || !generatedTokenIds.Any())
            {
                /*...*/
                return "";
            }

            Console.WriteLine(
                $"DEBUG: Trying to detokenize {generatedTokenIds.Count} tokens: [{string.Join(", ", generatedTokenIds)}]");
            try
            {
                string detokenizedText = tokenizer.Detokenize(generatedTokenIds.ToArray());
                if (string.IsNullOrWhiteSpace(detokenizedText))
                {
                    /*...*/
                }
                else
                {
                    Console.WriteLine($"DEBUG: Detokenized Text: >>>{detokenizedText}<<<");
                    return detokenizedText;
                }
            }
            catch (Exception ex)
            {
                /*...*/
            }
            finally
            {
                Console.WriteLine("--- END DEBUG ---");
            }

            return "";
        }

        // --- MÉTODO ProcessMessage CORRIGIDO ---
        public async Task ProcessMessage(WebSocket webSocket, string message)
        {
            if (string.IsNullOrEmpty(message)) return; // Verificação básica

            Console.WriteLine($"ChatBotService: === Processing message: '{message}' ===");

            Tensor? initialInputTensor = null;
            Tensor? currentInput = null;
            List<int> generatedTokenIds = new List<int>();
            string finalResponseMessage = "[Error: Generation failed]"; // Mensagem padrão inicial
            int lastPredictedTokenId = -1;

            try
            {
                // 1. Tokenizar Input Inicial
                int[] inputTokens = tokenizer.Tokenize(message); // Usa SharpToken
                int[] initialSequence = inputTokens.Where(t => t != this.padTokenId).ToArray();
                if (initialSequence.Length == 0)
                {
                    Console.WriteLine("ChatBotService: Input resulted in empty sequence after removing padding.");
                    await SendMessage(webSocket, "[Error: Invalid input]");
                    return;
                }

                Console.WriteLine($"ChatBotService: Initial non-pad sequence length: {initialSequence.Length}");

                // 2. Preparar Tensor Inicial
                long[] initialLongs = initialSequence.Select(id => (long)id).ToArray();
                initialInputTensor = tensor(initialLongs, dtype: ScalarType.Int64).to(device);
                currentInput = initialInputTensor.clone().to(device); // Clona para não modificar o original
                Console.WriteLine($"ChatBotService: Initial Input Tensor Shape: {currentInput.shape}");

                // 3. Loop de Geração
                model.eval(); // Garante modo de avaliação
                using (var noGrad = torch.no_grad()) // Desabilita gradientes
                {
                    for (int step = 0; step < this.maxGeneratedTokens; step++)
                    {
                        Console.WriteLine($"ChatBotService: --> Generation Step {step + 1}/{this.maxGeneratedTokens}");

                        Tensor? outputLogits = null;
                        Tensor? probabilities = null;
                        Tensor? predictedIndexTensor = null;
                        Tensor? scaledLogits = null;
                        int predictedTokenId = -1; // Mudei para int direto

                        try
                        {
                            // 3a. Forward Pass
                            outputLogits = model.forward(currentInput);
                            if ((bool)(outputLogits == null) || outputLogits.numel() == 0)
                            {
                                Console.Error.WriteLine("Error: Model returned null or empty logits.");
                                break;
                            }

                            // 3b. Penalidade de Repetição
                            if (lastPredictedTokenId != -1 && lastPredictedTokenId >= 0 &&
                                lastPredictedTokenId < outputLogits.shape[0]) // Verifica limites
                            {
                                outputLogits[lastPredictedTokenId] = -float.MaxValue;
                            }

                            // 3c. Sampling (Temperatura + Softmax + Multinomial)
                            scaledLogits = outputLogits / Math.Max(this.samplingTemperature, 1e-6f);
                            probabilities = torch.softmax(scaledLogits, dim: 0); // dim 0 pois esperamos [vocabSize]
                            predictedIndexTensor = torch.multinomial(probabilities, num_samples: 1);
                            predictedTokenId = (int)predictedIndexTensor.item<long>(); // Converte para int
                            Console.WriteLine($"ChatBotService: Step {step + 1}: Sampled Token ID: {predictedTokenId}");

                            // 3d. Verificar EOS
                            // *** Usa o HashSet eosTokenIds que agora só tem 50256 ***
                            if (this.eosTokenIds.Contains(predictedTokenId))
                            {
                                Console.WriteLine(
                                    $"ChatBotService: Step {step + 1}: EOS token ({predictedTokenId}) sampled. Stopping generation.");
                                break; // Sai do loop for
                            }

                            // 3e. Adicionar token gerado
                            generatedTokenIds.Add(predictedTokenId);
                            lastPredictedTokenId = predictedTokenId; // Guarda para próxima penalidade

                            // 3f. Preparar Próximo Input
                            var nextInputTensor = tensor(new long[] { (long)predictedTokenId }, dtype: ScalarType.Int64)
                                .to(device); // Tensor SÓ com o novo token

                            // Cria a nova sequência completa (ainda na CPU para facilitar)
                            var previousInputData = currentInput.to(CPU).data<long>().ToArray();
                            var nextSequenceLongs = previousInputData.Concat(new long[] { (long)predictedTokenId })
                                .ToArray();

                            // Limpa o tensor antigo do device
                            currentInput.Dispose();

                            // Limita ao tamanho máximo da sequência
                            if (nextSequenceLongs.Length > tokenizer.GetMaxSequenceLength())
                            {
                                int startIndex = nextSequenceLongs.Length - tokenizer.GetMaxSequenceLength();
                                // Pega os últimos 'maxSequenceLength' elementos
                                nextSequenceLongs = nextSequenceLongs.Skip(startIndex).ToArray();
                                // Console.WriteLine($"DEBUG: Input sequence truncated to last {tokenizer.GetMaxSequenceLength()} tokens.");
                            }

                            // Cria o tensor final para o próximo input no device correto
                            currentInput = tensor(nextSequenceLongs, dtype: ScalarType.Int64).to(device);

                            // Dispose do tensor intermediário (só o novo token)
                            nextInputTensor.Dispose();
                        }
                        catch (Exception stepEx)
                        {
                            Console.Error.WriteLine($"Error in generation step {step + 1}: {stepEx.ToString()}");
                            break; // Sai do loop se um passo falhar
                        }
                        finally
                        {
                            // Dispose dos tensores criados *dentro* do try do passo
                            outputLogits?.Dispose();
                            probabilities?.Dispose();
                            predictedIndexTensor?.Dispose();
                            scaledLogits?.Dispose(); // Dispose scaledLogits também
                        }
                    } // --- Fim do Loop FOR de Geração ---
                } // --- Fim do using no_grad ---

                Console.WriteLine(
                    $"ChatBotService: Generation loop finished. Generated {generatedTokenIds.Count} tokens.");

                // --- Detokenização e Refinamento (ESTRUTURA CORRIGIDA) ---
                if (generatedTokenIds.Any())
                {
                    try
                    {
                        // 1. Detokenize
                        string rawResponse = tokenizer.Detokenize(generatedTokenIds.ToArray()); // Usa SharpToken Decode
                        Console.WriteLine($"DEBUG: Raw detokenized response: >>>{rawResponse}<<<");

                        // 2. Refine
                        // *** CHAMA A FUNÇÃO QUE SERÁ ADICIONADA ABAIXO ***
                        finalResponseMessage = RefineGeneratedResponse(rawResponse);
                        Console.WriteLine($"DEBUG: Refined response: >>>{finalResponseMessage}<<<");

                        // 3. Verifica se ficou vazio após refinar
                        if (string.IsNullOrWhiteSpace(finalResponseMessage))
                        {
                            Console.WriteLine(
                                "ChatBotService: Final response message is empty or whitespace after refining. Sending default.");
                            finalResponseMessage = "[No meaningful response generated]";
                        }
                    }
                    catch (Exception dtEx) // Captura erros de Detokenize ou Refine
                    {
                        Console.Error.WriteLine($"Error during Detokenization or Refining: {dtEx.ToString()}");
                        finalResponseMessage = "[Error processing response]";
                    }
                }
                else // Caso nenhum token tenha sido gerado
                {
                    Console.WriteLine("ChatBotService: No tokens were generated.");
                    finalResponseMessage = "[No response generated]";
                }
                // --- Fim Detokenização e Refinamento ---

                // 4. Envia a resposta final
                await SendMessage(webSocket, finalResponseMessage);
                Console.WriteLine(
                    $"ChatBotService: SendMessage task awaited for final response: '{finalResponseMessage}'");
            } // Fim do Try Principal
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Error in ProcessMessage: {ex.ToString()}");
                try
                {
                    await SendMessage(webSocket, "[An internal error occurred]");
                }
                catch
                {
                    /* Ignora */
                }
            }
            finally
            {
                // Dispose seguro dos tensores principais
                initialInputTensor?.Dispose();
                currentInput?.Dispose();
            }
        } // --- Fim do ProcessMessage ---


        // *** ADICIONE ESTA FUNÇÃO AQUI DENTRO DA CLASSE ChatBotService ***
        private string RefineGeneratedResponse(string rawResponse)
        {
            if (string.IsNullOrWhiteSpace(rawResponse))
            {
                /*...*/
                return "[No response generated]";
            }

            Console.WriteLine($"DEBUG: Refining raw response: >>>{rawResponse}<<<");

            char[] sentenceEndingsStrict = { '.', '?'};
            string[] paragraphEndings = { "\n\n", "\r\n\r\n" };
            int earliestEnding = -1;
            bool endingIsStrict = false;
            char endingChar = '\0'; // Guarda o caractere finalizador

            // 1. Procura por ., !, ?
            int strictEnd = rawResponse.IndexOfAny(sentenceEndingsStrict);
            if (strictEnd != -1)
            {
                earliestEnding = strictEnd;
                endingIsStrict = true;
                endingChar = rawResponse[strictEnd]; // Guarda o caractere
                Console.WriteLine($"DEBUG: Found strict sentence ending '{endingChar}' at pos {strictEnd}.");
            }

            // 2. Procura por \n\n (MENOS prioritário)
            foreach (string pEnd in paragraphEndings)
            {
                int paraEnd = rawResponse.IndexOf(pEnd);
                if (paraEnd != -1 && (earliestEnding == -1 || paraEnd < earliestEnding))
                {
                    earliestEnding = paraEnd;
                    endingIsStrict = false; // Não é pontuação
                    endingChar = '\n'; // Marca como quebra
                    Console.WriteLine($"DEBUG: Found paragraph ending at pos {earliestEnding}.");
                }
            }

            // 3. Procura por \n (MENOR prioridade)
            int singleNewline = rawResponse.IndexOf('\n');
            if (singleNewline != -1 && (earliestEnding == -1 || singleNewline < earliestEnding))
            {
                earliestEnding = singleNewline;
                endingIsStrict = false;
                endingChar = '\n';
                Console.WriteLine($"DEBUG: Found single newline at pos {earliestEnding}.");
            }

            string refinedResponse;
            if (earliestEnding >= 0) // Encontrou algum finalizador
            {
                int cutPosition = earliestEnding; // Posição padrão é ANTES do marcador

                if (endingIsStrict) // Se terminou com ., !, ?
                {
                    cutPosition++; // Inclui a pontuação por padrão

                    // Lógica específica para ". " e "? "
                    if ((endingChar == '.' || endingChar == '?') &&
                        (earliestEnding + 1 < rawResponse.Length && rawResponse[earliestEnding + 1] == ' '))
                    {
                        // Verifica se NÃO é seguido por novo parágrafo
                        bool followedByParagraph = false;
                        // ... (lógica para verificar paragraphEndings após o espaço - como antes) ...
                        foreach (string pEnd in paragraphEndings)
                        {
                            if (rawResponse.Length >= earliestEnding + 2 + pEnd.Length &&
                                rawResponse.Substring(earliestEnding + 2).StartsWith(pEnd))
                            {
                                followedByParagraph = true;
                                break;
                            }
                        }

                        if (!followedByParagraph)
                        {
                            // É ". " ou "? " normal, corta após a pontuação (cutPosition já é +1)
                            Console.WriteLine(
                                $"DEBUG: Found '. ' or '? ' not followed by paragraph. Cutting after punctuation at {earliestEnding}.");
                        }
                        else
                        {
                            // Seguido por parágrafo. A lógica atual já cortará APÓS a pontuação.
                            // Se quiséssemos permitir múltiplos parágrafos, seria mais complexo.
                            Console.WriteLine(
                                $"DEBUG: Found '. ' or '? ' followed by paragraph. Still cutting after punctuation ({earliestEnding}) for now.");
                        }
                    }
                    else
                    {
                        Console.WriteLine(
                            $"DEBUG: Found strict ending '{endingChar}' (not '. ' or '? '). Cutting after punctuation at {earliestEnding}.");
                    }

                    refinedResponse = rawResponse.Substring(0, cutPosition).Trim();
                    Console.WriteLine($"DEBUG: Truncated response (strict ending) at cut position {cutPosition}.");
                }
                else // O corte foi por \n ou \n\n
                {
                    // cutPosition já está correto (no início da quebra)
                    refinedResponse = rawResponse.Substring(0, cutPosition).Trim();
                    Console.WriteLine($"DEBUG: Truncated response (newline/paragraph) at cut position {cutPosition}.");
                }
            }
            else
            {
                // Se não encontrou fim claro, usa a resposta crua
                refinedResponse = rawResponse.Trim();
                Console.WriteLine($"DEBUG: No clear ending marker found, using raw response.");
            }

            if (string.IsNullOrWhiteSpace(refinedResponse))
            {
                Console.WriteLine($"DEBUG: Refined response became empty, returning original raw trimmed.");
                return rawResponse.Trim();
            }

            return refinedResponse;
        }
        // *** FIM DA FUNÇÃO RefineGeneratedResponse ***


        // --- Função auxiliar SendMessage ---
        private async Task SendMessage(WebSocket webSocket, string message)
        {
            if (webSocket.State == WebSocketState.Open)
            {
                Console.WriteLine($"ChatBotService: Sending message: '{message}'"); // Log do envio
                var responseBuffer = Encoding.UTF8.GetBytes(message);
                try
                {
                    await webSocket.SendAsync(new ArraySegment<byte>(responseBuffer, 0, responseBuffer.Length),
                        WebSocketMessageType.Text,
                        true, // End of message
                        CancellationToken.None);
                }
                catch (WebSocketException wsEx)
                {
                    Console.Error.WriteLine(
                        $"Error sending WebSocket message: {wsEx.Message} (WebSocketErrorCode: {wsEx.WebSocketErrorCode}, NativeErrorCode: {wsEx.NativeErrorCode})");
                    // Tentar fechar o socket se o envio falhar por conexão ruim?
                    // if (webSocket.State == WebSocketState.Open || webSocket.State == WebSocketState.CloseSent) {
                    //     await webSocket.CloseAsync(WebSocketCloseStatus.InternalServerError, "Send error", CancellationToken.None);
                    // }
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"Generic error sending WebSocket message: {ex.Message}");
                }
            }
            else
            {
                Console.WriteLine(
                    $"ChatBotService: WebSocket no longer open (State: {webSocket.State}). Cannot send message.");
            }
        }
        // --- FIM SendMessage ---
    } // --- FIM ChatBotService ---

    // *** REMOVA QUALQUER DEFINIÇÃO "partial class Tokenizer" DESTE ARQUIVO ***
} // --- FIM Namespace ---