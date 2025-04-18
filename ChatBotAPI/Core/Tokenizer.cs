// Tokenizer.cs - AJUSTADO PARA tokenizer.json com <unk> ID 0 e ~50k tokens

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ChatBotAPI.Core
{
    public partial class Tokenizer
    {
        private readonly Dictionary<string, int> wordToIndex;
        private readonly Dictionary<int, string> indexToWord;
        public int PadTokenId { get; private set; }
        public int UnkTokenId { get; private set; }
        private readonly string unkToken = "<unk>"; // Token UNK padrão neste JSON
        private readonly string padToken; // Será definido como <unk>
        private readonly int maxSequenceLength;
        public readonly int vocabSizeLimit; // Limite configurado

        public Tokenizer(Dictionary<string, int>? loadedVocab, int maxSequenceLength, int vocabSizeLimit)
        {
            this.maxSequenceLength = maxSequenceLength;
            this.vocabSizeLimit = vocabSizeLimit;
            this.wordToIndex = new Dictionary<string, int>();
            this.indexToWord = new Dictionary<int, string>(); // Key: ID, Value: Word

            if (loadedVocab == null || loadedVocab.Count == 0) { /* ... throw ... */ }

            Console.WriteLine($"Initializing Tokenizer with {loadedVocab.Count} entries from loaded vocab.");
            int foundUnkTokenId = -1;

            // 1. Carrega TODOS os tokens do JSON usando TryAdd
            foreach (var kvp in loadedVocab)
            {
                string word = kvp.Key;
                int id = kvp.Value;

                if (wordToIndex.Count >= this.vocabSizeLimit && !wordToIndex.ContainsKey(word)) {
                    Console.WriteLine($"Warning: Vocab limit ({this.vocabSizeLimit}) reached. Skipping token '{word}'.");
                    continue;
                }

                // *** USA TryAdd PARA EVITAR ERRO DE CHAVE DUPLICADA ***
                bool addedWord = wordToIndex.TryAdd(word, id);
                bool addedIndex = indexToWord.TryAdd(id, word); // Tenta adicionar ID -> Palavra

                // Se o ID já existia, loga um aviso (não deveria acontecer com JSONs válidos)
                if (!addedIndex && indexToWord.TryGetValue(id, out var existingWord)) {
                     if (existingWord != word) { // Verifica se a palavra associada era diferente
                          Console.WriteLine($"Warning: Token ID {id} already exists in indexToWord with word '{existingWord}'. Skipping new word '{word}'. Possible duplicate ID in JSON?");
                     }
                     // Se a palavra for a mesma, TryAdd falhou mas não há problema real.
                }
                if (!addedWord) {
                     // Não deveria acontecer se a verificação ContainsKey acima for usada, mas como segurança:
                     Console.WriteLine($"Warning: Word '{word}' already exists in wordToIndex. Skipping duplicate add.");
                }


                // Guarda o ID do token UNK esperado
                if (word == this.unkToken) {
                    foundUnkTokenId = id;
                }
            }

             // 2. Define PAD e UNK usando o token <unk> (ID 0)
            if (foundUnkTokenId != -1) {
                // ... (lógica para definir PadTokenId e UnkTokenId como foundUnkTokenId) ...
                this.UnkTokenId = foundUnkTokenId;
                this.PadTokenId = foundUnkTokenId; // Usando o mesmo ID de UNK para PAD
                this.padToken = this.unkToken;
                Console.WriteLine($"Using token '{this.unkToken}' (ID: {this.UnkTokenId}) for both UNK and PAD.");
                 // Garante que o ID 0 esteja mapeado corretamente, mesmo que TryAdd tenha falhado antes
                 if (!indexToWord.ContainsKey(this.PadTokenId)) {
                     indexToWord.Add(this.PadTokenId, this.padToken);
                     Console.WriteLine($"Ensured PadTokenId {this.PadTokenId} maps to '{this.padToken}'.");
                 }

            } else {
                 // ... (lógica de fallback se <unk> não for encontrado) ...
                  Console.Error.WriteLine($"CRITICAL WARNING: Token '{this.unkToken}' not found! Using fallback ID 0.");
                  this.UnkTokenId = 0;
                  this.PadTokenId = 0;
                  this.padToken = "<PAD_FALLBACK>";
                  // Tenta adicionar <unk> e <PAD_FALLBACK> com ID 0 se ainda não existir
                  if (indexToWord.TryAdd(this.PadTokenId, this.padToken)) { // Tenta adicionar PAD
                       wordToIndex.TryAdd(this.padToken, this.PadTokenId);
                  }
                   if (wordToIndex.TryAdd(this.unkToken, this.UnkTokenId)) { // Tenta adicionar UNK
                        // Se adicionou UNK mas não PAD, garante que ID 0 mapeie para PAD no indexToWord
                        if(!indexToWord.ContainsKey(this.UnkTokenId)) {
                            indexToWord.Add(this.UnkTokenId, this.padToken); // Prioriza PAD para ID 0 se ambos falharam
                        } else if (indexToWord[this.UnkTokenId] != this.padToken){
                             Console.WriteLine($"Warning: Fallback added UNK '{this.unkToken}' to ID {this.UnkTokenId}, but could not map PAD to it.");
                        }
                   }
            }

            // 3. Define o tamanho real do vocabulário
            this.ActualVocabSize = wordToIndex.Count;
            Console.WriteLine($"Tokenizer initialized. Actual Vocab Size: {this.ActualVocabSize}, PadTokenId = {this.PadTokenId}, UnkTokenId = {this.UnkTokenId}");
            // ... (verificação final de tamanho) ...
        }

        // Propriedade para saber o tamanho real do vocabulário usado
        public int ActualVocabSize { get; private set; }
        
        // Tokenize e Detokenize podem permanecer os mesmos da versão anterior
        // que usava o vocabulário grande, pois eles usam os PadTokenId/UnkTokenId definidos
        // e fazem split por espaço (que ainda é uma simplificação do BPE).
        public int[] Tokenize(string text)
        {
             Console.WriteLine($"Warning: Using simplified space-based tokenization, not full BPE for '{text.Substring(0, Math.Min(30,text.Length))}'.");
             string processedText = text.ToLowerInvariant(); // RoBERTa/BPE costuma ser case-sensitive, mas mantemos ToLower por simplicidade aqui
             string[] words = processedText.Split(new[] { ' ', '\t', '\n', '\r'}, StringSplitOptions.RemoveEmptyEntries);
             List<int> tokens = new List<int>();

            foreach (string word in words)
            {
                if (wordToIndex.TryGetValue(word, out int index)) {
                    tokens.Add(index);
                } else if (wordToIndex.TryGetValue("Ġ" + word, out index)) { // Tentativa de prefixo de espaço BPE
                     tokens.Add(index);
                } else {
                    tokens.Add(this.UnkTokenId); // ID 0 neste caso
                }
            }
            // Padding / Truncating
             int currentLength = tokens.Count;
            if (currentLength < maxSequenceLength) {
                tokens.AddRange(Enumerable.Repeat(this.PadTokenId, maxSequenceLength - currentLength)); // Usa ID 0
            } else if (currentLength > maxSequenceLength) {
                tokens = tokens.GetRange(0, maxSequenceLength);
            }
            return tokens.ToArray();
        }

        public string Detokenize(int[] tokens)
        {
            List<string> words = new List<string>();
            foreach (int token in tokens)
            {
                // Ignora PAD/UNK (ID 0)
                if (token != this.PadTokenId && indexToWord.TryGetValue(token, out string? word))
                {
                    // Remove marcadores BPE conhecidos (pode precisar de mais)
                    word = word.Replace("Ġ", "").Replace("</w>", ""); // Tenta limpar
                    if (!string.IsNullOrEmpty(word)) // Adiciona apenas se não ficar vazio após limpar
                    {
                        words.Add(word);
                    }
                }
            }
            // *** MUDANÇA: Tentar juntar com espaço ***
            return string.Join(" ", words);
        }
        public bool TryGetTokenId(string token, out int id)
        {
            // Você pode querer normalizar o token aqui também, assim como faz em Tokenize
            // Mas para os tokens simples como ".", "?", "!", toLowerInvariant() deve bastar
            string processedToken = token.ToLowerInvariant();
            return wordToIndex.TryGetValue(processedToken, out id);
        }

         // Getter para MaxSequenceLength
        public int GetMaxSequenceLength()
        {
            return this.maxSequenceLength;
        }
    }
}