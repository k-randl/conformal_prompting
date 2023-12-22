import os
import numpy as np

from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer

from typing import Dict, Any 

####################################################################################################
# Tokenizer Class:                                                                                 #
####################################################################################################

class WordTokenizer:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokenizer = TreebankWordTokenizer()
        self._stemmer = PorterStemmer()
        self._vocab = ['[???]']
        self._train = True
        self._ignored_arguments = []

    @property
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        return len(self._vocab)
    
    @property
    def train(self) -> bool:
        """
        `bool`: Indicates whether training of the Tokenizer is active.
        """
        return self._train
    
    @train.setter
    def train(self, value:bool) -> None:
        """
        `bool`: Indicates whether training of the Tokenizer is active.
        """
        self._train = value

    def __call__(self, text:str, return_offsets_mapping:bool=True, **kwargs) -> Dict[str, Any]:
        """ Converts a string in a sequence of tokens (string), using the tokenizer. Split in words for word-based
        vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces).

        Do NOT take care of added tokens.
        """
        for argument in kwargs:
            if argument not in self._ignored_arguments:
                print(f'WARNING: ignoring argument "{argument}" in call to WordTokenizer.__call__(...)!')
                self._ignored_arguments.append(argument)

        offset_mapping = np.array([(i,j) for i,j in self._tokenizer.span_tokenize(text)], dtype=int)

        result =  {'input_ids': np.array([self._convert_token_to_id(text[i:j]) for i,j in offset_mapping], dtype=int)}

        if return_offsets_mapping:
            result['offset_mapping'] = offset_mapping

        return result

    def _convert_token_to_id(self, token:str) -> int:
        token = self._stemmer.stem(token, to_lowercase=True)
        try: return self._vocab.index(token)
        except ValueError:
            if self._train:
                self._vocab.append(token)
                return len(self._vocab) - 1
            else: return 0

    def _convert_id_to_token(self, index:int) -> str:
        try: return self._vocab[index]
        except IndexError: return self._vocab[0]

    @staticmethod
    def load(dir:str, **kwargs) -> 'WordTokenizer':
        # create WordTokenizer-object:
        tokenizer = WordTokenizer(**kwargs)

        # load vocab from file:
        with open(dir + '/vocab.pickle', 'r') as file:
            tokenizer._vocab = file.read().split('\n')
            tokenizer._train = False

        return tokenizer

    def save(self, dir:str) -> None:
        # create directories:
        os.makedirs(dir, exist_ok=True)

        # save vocabulary:
        with open(dir + '/vocab.pickle', 'w') as file:
            file.write('\n'.join(self._vocab))