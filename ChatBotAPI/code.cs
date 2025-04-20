// Coloque #!csharp no TOPO da célula para indicar código C#
#!csharp

using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim; // Adicionar para Adam
using SharpToken; // Para o tokenizer

// Teste básico
Console.WriteLine("Hello from C# in Colab!");
var tensor = torch.randn(2, 3);
Console.WriteLine(tensor.ToString());

// Verificar disponibilidade da GPU (deve funcionar se o runtime estiver como GPU)
var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
Console.WriteLine($"Using device: {device}");