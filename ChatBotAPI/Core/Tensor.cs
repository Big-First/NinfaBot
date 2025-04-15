using System.Text;

namespace ChatBotAPI.Core;

/// <summary>
/// Representa um Tensor multi-dimensional básico para fins de estudo.
/// Armazena dados como um array 1D (float[]) e usa strides para acesso.
/// </summary>
public class Tensor
{
    public readonly float[] _data; // Armazenamento 1D dos dados
    private readonly int[] _shape;  // Dimensões do tensor (imutável após criação)
    private readonly int[] _strides; // Passos para cálculo de índice (imutável após criação)

    /// <summary>
    /// Obtém o formato (dimensões) do tensor.
    /// </summary>
    public IReadOnlyList<int> Shape { get; } // Interface ReadOnly para proteger _shape

    /// <summary>
    /// Obtém o Rank (número de dimensões) do tensor.
    /// </summary>
    public int Rank => _shape.Length;

    /// <summary>
    /// Obtém o número total de elementos no tensor.
    /// </summary>
    public int Size { get; }

    /// <summary>
    /// Cria um tensor com o formato especificado, inicializado com zeros.
    /// </summary>
    /// <param name="shape">As dimensões do tensor. Cada dimensão deve ser positiva.</param>
    public Tensor(params int[] shape)
    {
        if (shape == null || shape.Length == 0)
            throw new ArgumentException("O formato (shape) não pode ser nulo ou vazio.", nameof(shape));
        if (shape.Any(d => d <= 0))
            throw new ArgumentException("Todas as dimensões do formato devem ser positivas.", nameof(shape));

        _shape = (int[])shape.Clone(); // Copia defensiva
        Shape = Array.AsReadOnly(_shape); // Expor como ReadOnly

        Size = CalculateSize(_shape);
        _data = new float[Size]; // Inicializa com zeros por padrão

        _strides = CalculateStrides(_shape);
    }

    /// <summary>
    /// Cria um tensor com o formato e dados iniciais especificados.
    /// Os dados são fornecidos como um array 1D (flat) na ordem C (row-major).
    /// </summary>
    /// <param name="data">Array 1D contendo os dados iniciais.</param>
    /// <param name="shape">As dimensões do tensor.</param>
    public Tensor(float[] data, params int[] shape)
    {
        if (shape == null || shape.Length == 0)
            throw new ArgumentException("O formato (shape) não pode ser nulo ou vazio.", nameof(shape));
        if (shape.Any(d => d <= 0))
            throw new ArgumentException("Todas as dimensões do formato devem ser positivas.", nameof(shape));
        if (data == null)
            throw new ArgumentNullException(nameof(data));

        _shape = (int[])shape.Clone();
        Shape = Array.AsReadOnly(_shape);
        Size = CalculateSize(_shape);

        if (data.Length != Size)
        {
            throw new ArgumentException($"O tamanho dos dados ({data.Length}) deve corresponder ao tamanho total calculado pelo formato ({Size}).");
        }

        _data = (float[])data.Clone(); // Copia defensiva dos dados
        _strides = CalculateStrides(_shape);
    }

    // -- Métodos Auxiliares Internos --

    private static int CalculateSize(int[] shape)
    {
        // Calcula o número total de elementos multiplicando as dimensões
        int size = 1;
        foreach (int dim in shape)
        {
            size *= dim;
        }
        return size;
    }

    private static int[] CalculateStrides(int[] shape)
    {
        // Calcula os strides para cada dimensão
        int rank = shape.Length;
        int[] strides = new int[rank];
        if (rank == 0) return strides; // Tensor escalar (embora evitado no construtor atual)

        int currentStride = 1;
        for (int i = rank - 1; i >= 0; i--)
        {
            strides[i] = currentStride;
            currentStride *= shape[i];
        }
        // Inverte para a ordem correta (stride[i] = produto das dims > i)
        // Não, a lógica acima está ligeiramente errada para a definição padrão de stride.
        // Stride[i] é quantos elementos pular para avançar 1 na dimensão i.
        // Ex: Shape [2, 3, 4] -> Strides [12, 4, 1]
        // Stride[n-1] = 1
        // Stride[i] = Stride[i+1] * Shape[i+1]

        strides[rank - 1] = 1; // Stride da última dimensão é sempre 1
        for (int i = rank - 2; i >= 0; i--)
        {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        return strides;
    }

    private void ValidateIndices(int[] indices)
    {
        // Valida se os índices fornecidos são válidos para o formato do tensor
        if (indices == null)
            throw new ArgumentNullException(nameof(indices));
        if (indices.Length != Rank)
            throw new ArgumentException($"Esperado {Rank} índices, mas foram fornecidos {indices.Length}.", nameof(indices));

        for (int i = 0; i < Rank; i++)
        {
            if (indices[i] < 0 || indices[i] >= _shape[i])
            {
                throw new IndexOutOfRangeException($"Índice {indices[i]} está fora do intervalo para a dimensão {i} (tamanho {_shape[i]}).");
            }
        }
    }

    private int GetFlatIndex(params int[] indices)
    {
        // Calcula o índice do array 1D (_data) a partir dos índices multi-dimensionais
        // Assume que os índices já foram validados
        int flatIndex = 0;
        for (int i = 0; i < Rank; i++)
        {
            flatIndex += indices[i] * _strides[i];
        }
        return flatIndex;
    }

    // -- Métodos Públicos e Indexador --

    /// <summary>
    /// Obtém o valor no tensor nos índices especificados.
    /// </summary>
    /// <param name="indices">Os índices multi-dimensionais.</param>
    /// <returns>O valor float no local especificado.</returns>
    public float Get(params int[] indices)
    {
        ValidateIndices(indices);
        int flatIndex = GetFlatIndex(indices);
        return _data[flatIndex];
    }

    /// <summary>
    /// Define o valor no tensor nos índices especificados.
    /// </summary>
    /// <param name="value">O valor float a ser definido.</param>
    /// <param name="indices">Os índices multi-dimensionais.</param>
    public void Set(float value, params int[] indices)
    {
        ValidateIndices(indices);
        int flatIndex = GetFlatIndex(indices);
        _data[flatIndex] = value;
    }

    /// <summary>
    /// Acessa ou modifica o valor no tensor usando índices multi-dimensionais.
    /// Ex: tensor[1, 0, 2]
    /// </summary>
    /// <param name="indices">Os índices multi-dimensionais.</param>
    public float this[params int[] indices]
    {
        get => Get(indices);
        set => Set(value, indices);
    }

    /// <summary>
    /// Retorna uma representação de string do tensor (pode ser grande!).
    /// </summary>
    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.Append($"Tensor(Shape: [{string.Join(", ", _shape)}], DataType: float)\n");
        AppendDataString(sb, 0, new int[Rank]); // Chama helper recursivo
        return sb.ToString();
    }

    // Helper recursivo para construir a string formatada
    private void AppendDataString(StringBuilder sb, int dimension, int[] currentIndices)
    {
        if (dimension == Rank) // Caso base: chegou a um elemento individual
        {
            sb.Append(this[currentIndices].ToString("G7")); // Formato geral com precisão
        }
        else // Caso recursivo: processa a dimensão atual
        {
            sb.Append('[');
            for (int i = 0; i < Shape[dimension]; i++)
            {
                currentIndices[dimension] = i; // Define o índice para a dimensão atual
                AppendDataString(sb, dimension + 1, currentIndices); // Chama recursivamente para a próxima dimensão

                if (i < Shape[dimension] - 1) // Adiciona separador se não for o último elemento
                {
                    sb.Append(", ");
                    if (dimension < Rank - 2) // Adiciona quebra de linha entre "blocos" de dimensões mais altas
                    {
                       // Pode adicionar indentação aqui para melhor formatação, mas complica
                       // sb.Append("\n" + new string(' ', dimension + 2));
                    }
                    else if (dimension == Rank - 2 && Rank > 1) // Quebra linha entre linhas de uma matriz 2D
                    {
                        sb.Append("\n "); // Adiciona espaço para alinhar
                    }
                }
            }
            sb.Append(']');
        }
    }

     // --- Métodos Estáticos de Fábrica (Exemplos) ---

    /// <summary>
    /// Cria um tensor preenchido com zeros.
    /// </summary>
    public static Tensor Zeros(params int[] shape)
    {
        return new Tensor(shape);
    }

    /// <summary>
    /// Cria um tensor preenchido com uns.
    /// </summary>
    public static Tensor Ones(params int[] shape)
    {
        var tensor = new Tensor(shape);
        for (int i = 0; i < tensor.Size; i++)
        {
            tensor._data[i] = 1.0f;
        }
        return tensor;
    }

    /// <summary>
    /// Cria um tensor a partir de um array 1D e um formato.
    /// </summary>
    public static Tensor FromData(float[] data, params int[] shape)
    {
        return new Tensor(data, shape);
    }

    // --- Operações Básicas (Exemplo: Adição Element-wise) ---

    /// <summary>
    /// Adiciona outro tensor a este tensor, elemento por elemento.
    /// Os tensores devem ter o mesmo formato.
    /// </summary>
    /// <param name="other">O outro tensor a ser adicionado.</param>
    /// <returns>Um novo tensor com o resultado da adição.</returns>
    public Tensor Add(Tensor other)
    {
        if (!this.Shape.SequenceEqual(other.Shape))
            throw new ArgumentException("Os tensores devem ter o mesmo formato para adição element-wise.");

        var resultData = new float[this.Size];
        for (int i = 0; i < this.Size; i++)
        {
            resultData[i] = this._data[i] + other._data[i];
        }
        // Cria um novo tensor com os dados resultantes e o mesmo formato
        return new Tensor(resultData, this._shape);
    }

    // Sobrecarga do operador + para conveniência
    public static Tensor operator +(Tensor a, Tensor b)
    {
       if (a is null) throw new ArgumentNullException(nameof(a));
       if (b is null) throw new ArgumentNullException(nameof(b));
       return a.Add(b);
    }

    // TODO: Implementar outras operações (subtração, multiplicação element-wise,
    // multiplicação de matrizes, reshape, slicing, etc.)
}