import tiktoken
import sys

try:
    enc = tiktoken.encoding_for_model("gpt2")
except ImportError:
    print("Erro: A biblioteca 'tiktoken' não está instalada. Execute 'pip install tiktoken'.")
    sys.exit(1)

def encode(text):
    """Tokeniza o texto usando tiktoken e retorna uma string com os IDs."""
    return " ".join(map(str, enc.encode(text)))

def decode(tokens_str):
    """Detokeniza uma string de IDs separados por espaço usando tiktoken."""
    if not tokens_str.strip():
        return ""
    try:
        token_ids = [int(x) for x in tokens_str.split()]
        return enc.decode(token_ids)
    except ValueError:
        return "Erro: Formato inválido para decodificação."

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python tokenizer.py <encode|decode> <texto ou tokens>")
        sys.exit(1)

    command = sys.argv[1].lower()
    data = " ".join(sys.argv[2:])

    if command == "encode":
        print(encode(data))
    elif command == "decode":
        print(decode(data))
    else:
        print("Erro: Comando inválido. Use 'encode' ou 'decode'.")