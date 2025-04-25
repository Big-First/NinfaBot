# core/tokenizer_py.py

import tiktoken
from typing import List, Set, Dict, Optional
import sys  # Para Exception


class Tokenizer:

    def __init__(self, encoding_name: str = "gpt2"):
        try:
            self._encoding = tiktoken.get_encoding(encoding_name)

            # Para o encoding 'gpt2', <|endoftext|> é o principal token especial.
            # O ID para <|endoftext|> em gpt2 é 50256.
            self.EOS_ID = 50256
            self.PAD_ID = self.EOS_ID  # Usando EOS como PAD_ID, comum em LLMs
            # <|startoftext|> não existe no encoding gpt2 padrão. Vamos definir como -1.
            self.START_ID = -1
            self.UNK_ID = -1  # tiktoken não tem um conceito direto de UNK como um token único

            self._known_special_tokens: Dict[str, int] = {}
            self._known_special_tokens["<|endoftext|>"] = self.EOS_ID

            print("-" * 50)
            print(f"Tokenizer '{encoding_name}' inicializado.")
            print(f"  VocabSize: {self.GetVocabSize()}")
            print(f"  EOS_ID: {self.EOS_ID} ('<|endoftext|>')")
            print(f"  PAD_ID: {self.PAD_ID} (usando EOS_ID)")
            print(f"  START_ID: {self.START_ID} ({'Não encontrado' if self.START_ID == -1 else '<|startoftext|>'})")
            print("-" * 50)

            if self.EOS_ID == -1:
                print("ERRO: EOS Token ID (<|endoftext|>) não encontrado para este encoding!")

        except Exception as ex:
            print(f"ERRO fatal ao inicializar Tokenizer com encoding '{encoding_name}': {ex}")
            # import traceback; traceback.print_exc() # Opcional para debug
            raise RuntimeError(f"Falha ao inicializar Tokenizer: {ex}") from ex

    def Encode(self, text: str, allow_special_tokens_in_text: bool = True) -> List[int]:
        """
        Codifica texto em uma lista de IDs de token usando tiktoken.
        Args:
            text: O texto a ser codificado.
            allow_special_tokens_in_text: Se True, tokens especiais ENCONTRADOS NO TEXTO
                                         serão codificados como tokens únicos.
                                         Se False, serão tratados como texto comum.
        Returns:
            Lista de IDs de token.
        """
        if allow_special_tokens_in_text:
            allowed_special_set = "all"
        else:
            allowed_special_set = set()  # Set vazio para não permitir nenhum token especial literal

        # Se você remover a string literal <|endoftext|> do dataset (como recomendado),
        # allow_special_tokens_in_text=False no trainer é correto.
        # Se você precisar permitir alguns tokens especiais (como <|im_start|>) no input do usuário
        # mas não <|endoftext|>, a lógica precisaria ser mais complexa, usando disallowed_special.
        # Para o dataset de treino sem a string, allow_special_tokens_in_text=False está correto.

        return self._encoding.encode(text, allowed_special=allowed_special_set)

    def Decode(self, ids: List[int]) -> str:
        """
        Decodifica uma lista de IDs de token de volta para texto.
        """
        return self._encoding.decode(ids)

    def GetVocabSize(self) -> int:
        """
        Obtém o tamanho do vocabulário do tokenizer.
        """
        return self._encoding.n_vocab

    def GetEosTokenId(self) -> int:
        """
        Obtém o ID do token de fim de sequência (EOS).
        """
        if self.EOS_ID == -1:
            print("AVISO: GetEosTokenId chamado, mas EOS Token ID não foi encontrado durante a inicialização.")
        return self.EOS_ID

    def GetPadTokenId(self) -> int:
        """
         Obtém o ID do token de padding (PAD).
        """
        return self.PAD_ID

    def get_special_token_id(self, special_token_string: str) -> int:
        """
        Verifica se uma string corresponde a um token especial conhecido e retorna seu ID.
        Retorna -1 se não for um token especial conhecido.
        """
        return self._known_special_tokens.get(special_token_string, -1)