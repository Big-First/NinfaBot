using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace ChatBotAPI.Core
{
    public class BinaryTreeNeuralModel
    {
        private Node? _root; // Nó raiz da árvore/Trie
        private readonly string _modelFilePath;
        private readonly Tokenizer _tokenizer;
        private readonly Random _random = new Random();

        public BinaryTreeNeuralModel(Tokenizer tokenizer, string modelFilePath)
        {
            _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
            _modelFilePath = modelFilePath ?? throw new ArgumentNullException(nameof(modelFilePath));
            LoadModel();
        }

        public void Train(List<int> inputTokens, List<int> targetTokens)
        {
            if (inputTokens == null || !inputTokens.Any() || targetTokens == null || !targetTokens.Any())
            {
                Console.WriteLine("Aviso Treino: Dados de treinamento inválidos ou vazios. Pulando.");
                return;
            }

            // Limpa os tokens EndOfText (consistente com a lógica anterior)
            // NOTA: Isso impede o modelo de aprender a *prever* o fim do texto.
            var cleanedInput = inputTokens.Where(t => t != Tokenizer.EndOfTextTokenId).ToList();
            var cleanedTarget = targetTokens.Where(t => t != Tokenizer.EndOfTextTokenId).ToList();

            if (!cleanedInput.Any() || !cleanedTarget.Any())
            {
                Console.WriteLine("Aviso Treino: Dados vazios após remover EndOfTextToken. Pulando.");
                return;
            }

            Console.WriteLine($"Treinando com Entrada: [{string.Join(",", cleanedInput)}], Alvo: [{string.Join(",", cleanedTarget)}]");

            // Inicializa a raiz se a árvore estiver vazia
            if (_root == null)
            {
                _root = new Node { Token = cleanedInput[0] }; // Assume o primeiro token como raiz para a primeira vez
                Console.WriteLine($"Nó Raiz inicializado com Token: {cleanedInput[0]}");
            }

            Node current = _root;
            int depth = 0;
            const int maxDepth = 50; // Limite de segurança

            // Percorre/Constrói o caminho na árvore para a sequência de entrada
            foreach (var token in cleanedInput)
            {
                // Se o nó atual já representa o token (ex: primeiro token é a raiz),
                // apenas continue para o próximo token da entrada para encontrar/criar seu filho.
                if (current.Token == token && current == _root) // Caso especial para o primeiro token ser a raiz
                {
                     Console.WriteLine($"Treino - Iniciando na raiz com token {token}");
                     continue; // Não precisa descer na árvore ainda
                }

                depth++;
                if (depth > maxDepth)
                {
                    Console.WriteLine($"Aviso Treino: Profundidade máxima ({maxDepth}) alcançada. Parando esta sequência aqui.");
                    break;
                }

                Node? nextNode; // Nó para onde vamos avançar

                // **Lógica Refatorada: Usa Dicionário de Filhos**
                if (current.Filhos.TryGetValue(token, out Node? existingChild))
                {
                    // Nó filho para este token já existe, avança para ele
                    nextNode = existingChild;
                    Console.WriteLine($"Treino - Avançando para nó filho existente: {token}");
                }
                else
                {
                    // Nó filho para este token não existe, cria um novo
                    var newNode = new Node { Token = token };
                    current.Filhos.Add(token, newNode); // Adiciona ao dicionário do pai
                    nextNode = newNode;
                    Console.WriteLine($"Treino - Criado e avançando para novo nó filho: {token}");
                }
                current = nextNode; // Atualiza o nó atual
            }

            // Adiciona os tokens alvo à lista NextTokens do nó final da sequência de entrada
            if (current != null) // Garante que a travessia foi bem-sucedida
            {
                // current.NextTokens é inicializado no construtor de Node
                foreach (var targetToken in cleanedTarget)
                {
                    if (!current.NextTokens.Contains(targetToken))
                    {
                        current.NextTokens.Add(targetToken);
                        Console.WriteLine($"Adicionado Token Alvo {targetToken} às previsões do Nó {current.Token}");
                    }
                }
            }
            else
            {
                 Console.WriteLine("Erro Treino: Nó atual tornou-se nulo durante travessia. Não é possível adicionar tokens alvo.");
            }
            // REMOVIDO: SaveModel() não é mais chamado aqui. Será chamado pelo Trainer.
        }

        public int GenerateNextToken(List<int> inputTokens, List<int> generatedTokens)
        {
            Node? current = _root;

            if (current == null)
            {
                Console.WriteLine("Aviso Geração: Raiz do modelo é nula. Usando fallback.");
                return FallbackToken(generatedTokens);
            }

            var cleanedInput = inputTokens.Where(t => t != Tokenizer.EndOfTextTokenId).ToList();

            Console.WriteLine($"Gerando próximo token para Entrada Limpa: [{string.Join(",", cleanedInput)}]");

            if (!cleanedInput.Any())
            {
                Console.WriteLine("Aviso Geração: Entrada vazia. Usando previsões da raiz.");
                if (current.NextTokens != null && current.NextTokens.Any())
                {
                    return SelectTokenFromList(current.NextTokens, generatedTokens, "Raiz (Entrada Vazia)");
                }
                else
                {
                    Console.WriteLine("Aviso Geração: Raiz não tem previsões. Usando fallback.");
                    return FallbackToken(generatedTokens);
                }
            }

            int depth = 0;
            const int maxDepth = 50;
            bool pathFound = true; // Assume que encontraremos o caminho

            // Percorre a árvore seguindo a sequência de entrada
            foreach (var token in cleanedInput)
            {
                Console.WriteLine($"Geração - Nó Atual: {current?.Token.ToString() ?? "Nulo"}, Procurando por filho: {token}");

                if (current == null) { // Segurança extra, embora não deva acontecer com a lógica abaixo
                    pathFound = false;
                    Console.WriteLine("Erro Geração: Nó atual inesperadamente nulo.");
                    break;
                }

                 // Se o nó atual já é o token E é a raiz (caso do primeiro token), não tente descer
                 if (current.Token == token && current == _root) {
                     Console.WriteLine($"Geração - Iniciando na raiz com token {token}");
                     continue;
                 }


                depth++;
                if (depth > maxDepth)
                {
                    Console.WriteLine($"Aviso Geração: Profundidade máxima ({maxDepth}) alcançada. Usando nó atual.");
                    break; // Usa previsões do nó onde parou
                }

                // **Lógica Refatorada: Busca no Dicionário de Filhos**
                if (current.Filhos.TryGetValue(token, out Node? nextNode))
                {
                    // Encontrou o próximo nó na sequência
                    Console.WriteLine($"Geração - Encontrado filho {token}. Avançando para Nó {nextNode.Token}");
                    current = nextNode;
                }
                else
                {
                    // Não encontrou o próximo nó -> sequência de entrada diverge do que foi treinado
                    Console.WriteLine($"Aviso Geração: Token {token} não encontrado como filho do Nó {current.Token}. Sequência não encontrada na árvore.");
                    pathFound = false;
                    break; // Para a travessia
                }
            }

            // Seleciona o próximo token
            if (current == null || !pathFound)
            {
                // Se a travessia falhou em algum ponto
                Console.WriteLine("Aviso Geração: Falha ao encontrar o caminho completo da entrada na árvore. Usando fallback.");
                return FallbackToken(generatedTokens);
            }
            else if (current.NextTokens == null || !current.NextTokens.Any())
            {
                // Se chegou ao fim do caminho, mas este nó não tem previsões aprendidas
                Console.WriteLine($"Aviso Geração: Nó final {current.Token} não tem previsões (NextTokens). Usando fallback.");
                return FallbackToken(generatedTokens);
            }
            else
            {
                // Chegou ao fim do caminho e tem previsões, seleciona uma
                return SelectTokenFromList(current.NextTokens, generatedTokens, $"Nó {current.Token}");
            }
        }

        private int SelectTokenFromList(List<int> candidates, List<int> generatedTokens, string sourceNodeDesc)
        {
            if (!candidates.Any())
            {
                 Console.WriteLine($"Erro SelectToken: Lista de candidatos vazia para {sourceNodeDesc}. Usando fallback.");
                 return FallbackToken(generatedTokens);
            }
            int position = generatedTokens.Count % candidates.Count;
            int nextToken = candidates[position];
            Console.WriteLine($"Token {nextToken} selecionado de {sourceNodeDesc} (candidatos: [{string.Join(",", candidates)}], índice: {position}).");
            return nextToken;
        }

        // Mecanismo de fallback
        private int FallbackToken(List<int> generatedTokens)
        {
            Console.WriteLine("Executando fallback de geração de token.");
            // Implementação atual: token aleatório (fraco)
            var allTokens = _tokenizer.GetAllTokenIds();
            if (!allTokens.Any())
            {
                Console.WriteLine("Erro Fallback: Nenhum token disponível no vocabulário. Retornando -1.");
                return -1;
            }
            int randomToken = allTokens[_random.Next(allTokens.Count)];
            Console.WriteLine($"Fallback: Token aleatório {randomToken} selecionado.");
            return randomToken;
            // TODO: Considerar fallbacks melhores (ex: usar previsões da raiz se disponíveis)
        }

        // Carrega o modelo do arquivo JSON
        private void LoadModel()
        {
            if (File.Exists(_modelFilePath))
            {
                try
                {
                    Console.WriteLine($"Carregando modelo de {_modelFilePath}...");
                    var json = File.ReadAllText(_modelFilePath);
                    var options = new JsonSerializerOptions { IncludeFields = true }; // Pode ser necessário se Node usa campos públicos
                    _root = JsonSerializer.Deserialize<Node>(json); // Removido options, dicionário deve serializar bem
                    Console.WriteLine("Modelo carregado com sucesso.");
                     if (_root == null) {
                          Console.WriteLine("Aviso Carregamento: Modelo desserializado como nulo. Iniciando vazio.");
                     }
                }
                catch (Exception ex)
                {
                     Console.WriteLine($"Erro ao carregar modelo de {_modelFilePath}: {ex.Message}. Iniciando com um modelo vazio.");
                     _root = null;
                }
            }
            else
            {
                Console.WriteLine($"Arquivo do modelo não encontrado em {_modelFilePath}. Iniciando com um modelo vazio.");
                _root = null;
            }
        }

        // Salva o modelo atual no arquivo JSON (agora público para ser chamado pelo Trainer)
        public void SaveModel()
        {
            if (_root == null)
            {
                 Console.WriteLine("Aviso SaveModel: Raiz nula. Nada para salvar.");
                 return;
            }
            try
            {
                string? directory = Path.GetDirectoryName(_modelFilePath);
                if (directory != null && !Directory.Exists(directory))
                {
                    Directory.CreateDirectory(directory);
                }

                Console.WriteLine($"Salvando modelo em {_modelFilePath}...");
                var options = new JsonSerializerOptions { WriteIndented = true }; // Formatação
                var json = JsonSerializer.Serialize(_root, options);
                File.WriteAllText(_modelFilePath, json);
                Console.WriteLine("Modelo salvo com sucesso.");
            }
            catch (Exception ex)
            {
                 Console.WriteLine($"Erro ao salvar modelo em {_modelFilePath}: {ex.Message}");
            }
        }
    }
}