// Trainer.cs - AJUSTE PARA TESTE: Mover model.train() para DENTRO do foreach

using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics; // Para Stopwatch
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.optim; // Para Adam
using static TorchSharp.torch.nn;    // Para CrossEntropyLoss, Module

namespace ChatBotAPI.Core
{
    public class Trainer
    {
        private readonly TorchSharpModel model;
        private readonly Tokenizer tokenizer;
        private readonly Optimizer optimizer;
        private readonly Module<Tensor, Tensor, Tensor> lossFunction;
        private readonly Device device;
        private readonly string modelSavePath;

        // --- CONSTRUTOR (sem alterações) ---
        public Trainer(TorchSharpModel model, Tokenizer tokenizer, double learningRate, string modelSavePath) // <-- Remove parâmetro trainingState
        {
            this.model = model ?? throw new ArgumentNullException(nameof(model));
            this.tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
            this.device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            this.modelSavePath = Path.GetFullPath(modelSavePath);
            // this.trainingState = trainingState; // REMOVIDO
            Console.WriteLine($"Trainer using device: {this.device.type}");
            Console.WriteLine($"Trainer configured to save model to: {this.modelSavePath}");
            this.model.to(this.device);
            this.optimizer = Adam(this.model.parameters(), lr: learningRate);
            int padTokenId = this.tokenizer.PadTokenId;
            this.lossFunction = CrossEntropyLoss(ignore_index: padTokenId).to(this.device);
            Console.WriteLine($"Trainer initialized loss function. Device: {this.device.type}. ignore_index (PadTokenId): {padTokenId}");
            if (tokenizer.ActualVocabSize <= 2) {
                 Console.Error.WriteLine("CRITICAL WARNING: Trainer detected Tokenizer ActualVocabSize <= 2...");
             }
        }

        // --- MÉTODO Train (COM model.train() MOVIDO PARA TESTE) ---
        public void Train(List<string> trainingData, int epochs)
        {
            // Verificações iniciais
            if (trainingData == null || !trainingData.Any()) { Console.WriteLine("Training data is empty. Skipping training."); return; }
            if (tokenizer.ActualVocabSize <= 2) { Console.Error.WriteLine("Cannot train: Tokenizer vocabulary is too small (Size <= 2)."); return; }

            int padTokenId = tokenizer.PadTokenId;
            int vocabSize = tokenizer.ActualVocabSize;
            int maxSeqLen = 50; // Valor padrão, ajuste
             try { maxSeqLen = tokenizer.GetMaxSequenceLength(); }
             catch { Console.WriteLine($"Warning: Could not get MaxSequenceLength. Using fallback={maxSeqLen}."); }

            Console.WriteLine($"Starting TorchSharp training on {device.type}. Epochs: {epochs}. Sentences: {trainingData.Count}. PadTokenId: {padTokenId}. VocabSize: {vocabSize}. MaxSeqLen: {maxSeqLen}.");
            Stopwatch epochStopwatch = new Stopwatch();

            // --- Loop Principal de Épocas ---
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                // ***** model.train() FOI REMOVIDO DAQUI *****

                epochStopwatch.Restart();
                Console.WriteLine($"--- Epoch {epoch + 1}/{epochs} ---");
                float totalLoss = 0f;
                long totalStepsInEpoch = 0;
                long skippedSteps = 0;
                long sentenceCount = 0;

                Console.WriteLine($"DEBUG: Epoch {epoch + 1}: Entering sentence loop...");
                try
                {
                    // --- Loop pelas Sentenças ---
                    foreach (string sentence in trainingData)
                    {
                         // Log no início do corpo do foreach
                         // Console.WriteLine($"DEBUG: Epoch {epoch + 1}: INSIDE foreach - START body for sentenceCount = {sentenceCount}");
                         sentenceCount++;

                         // ***** Tentativa de chamar model.train() aqui dentro *****
                         try {
                             // Console.WriteLine($"DEBUG: Epoch {epoch+1}, Sent {sentenceCount}: Calling model.train() (Inside loop)..."); // Log Opcional
                             model.train(); // Chama train para cada sentença neste teste
                         } catch (Exception innerTrainEx) {
                              Console.Error.WriteLine($"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
                              Console.Error.WriteLine($"ERRO FATAL CHAMANDO model.train() DENTRO DO LOOP (Epoch {epoch + 1}, Sent {sentenceCount}): {innerTrainEx.Message}");
                              Console.Error.WriteLine($"Detalhes: {innerTrainEx.ToString()}");
                              Console.Error.WriteLine($"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
                              // Decide se quer parar ou apenas pular esta sentença
                              skippedSteps++;
                              continue; // Pula para a próxima sentença se train() falhar aqui
                         }
                         // ***** Fim chamada interna model.train() *****


                         // Tokenizar
                         int[] tokens = tokenizer.Tokenize(sentence);

                         // Verificar tamanho mínimo
                         if (tokens.Length <= 1) {
                             Console.WriteLine($"AVISO: Epoch {epoch + 1}, Sent {sentenceCount} -> tokens.Length <= 1. Skipping.");
                             skippedSteps++;
                             continue; // PULA PARA A PRÓXIMA SENTENÇA
                         }

                         // --- Loop Interno pelos Passos ---
                         for (int i = 1; i < tokens.Length; i++)
                         {
                             // Pular se for padding
                             if (tokens[i] == padTokenId) {
                                 skippedSteps++;
                                 continue; // PULA PARA O PRÓXIMO PASSO 'i'
                             }

                             // Preparar input/target
                             long[] inputSequence = tokens.Take(i).Select(t => (long)t).ToArray();
                             long targetTokenId = tokens[i];
                             if (!inputSequence.Any()) { skippedSteps++; continue; }

                             // Declarar tensores
                             Tensor? inputTensor = null;
                             Tensor? targetTensor = null;
                             Tensor? outputLogits = null;
                             Tensor? loss = null;

                             // Bloco TRY/CATCH/FINALLY para o Passo
                             try
                             {
                                 // Criar tensores
                                 inputTensor = tensor(inputSequence, dtype: ScalarType.Int64).to(device);
                                 targetTensor = tensor(new long[] { targetTokenId }, dtype: ScalarType.Int64).to(device);

                                 // Zerar gradientes
                                 optimizer.zero_grad();

                                 // Forward pass (Modelo já está em modo treino por causa da chamada no início do foreach)
                                 outputLogits = model.forward(inputTensor);

                                 // Verificar nulidade e shape da saída
                                 if ((bool)(outputLogits == null)) { /* ... log WARN e continue ... */ skippedSteps++; continue; }
                                 bool isShapeOk = (outputLogits.dim() == 1 && outputLogits.shape[0] == vocabSize);
                                 if (!isShapeOk) { /* ... log WARN e continue ... */ skippedSteps++; continue; }

                                 // Calcular Loss
                                 loss = lossFunction.forward(outputLogits!.unsqueeze(0), targetTensor!);
                                 float currentLoss = loss!.item<float>();

                                 // Backward e Step
                                 loss.backward();
                                 optimizer.step();

                                 // Acumular perda e contar passo válido
                                 totalLoss += currentLoss;
                                 totalStepsInEpoch++;

                             }
                             catch (Exception stepEx) { /* ... log ERRO FATAL ... */ skippedSteps++; }
                             finally { /* ... dispose tensores ... */ }
                             // --- Fim do Bloco TRY/CATCH/FINALLY do Passo ---

                         } // --- Fim do Loop Interno (Passos 'i') ---

                    } // --- Fim do Loop Externo (Sentenças 'foreach') ---

                    Console.WriteLine($"DEBUG: Epoch {epoch + 1}: Finished sentence loop.");
                }
                catch (Exception exOuterLoop) { /* ... log ERRO FATAL EXTERNO ... */ }

                // --- Fim da Época ---
                // ***** Colocar em modo eval no fim da época, já que treinamos dentro do loop *****
                 Console.WriteLine($"DEBUG: Epoch {epoch + 1}: Setting model.eval() at end of epoch.");
                 model.eval();
                // ***** Fim model.eval() *****

                epochStopwatch.Stop();
                float avgLoss = totalStepsInEpoch > 0 ? totalLoss / totalStepsInEpoch : 0f;
                Console.WriteLine($"--- Epoch {epoch + 1} completed in {epochStopwatch.ElapsedMilliseconds} ms. Avg Loss: {avgLoss:F6}, Steps: {totalStepsInEpoch}, Skipped: {skippedSteps} ---");

            } // --- Fim do Loop Principal de Épocas ---

            // model.eval(); // Chamado no fim de cada época agora
            Console.WriteLine("Training finished.");
        } // --- Fim do Método Train ---
    } // --- Fim da Classe Trainer ---
} // --- Fim do Namespace ---