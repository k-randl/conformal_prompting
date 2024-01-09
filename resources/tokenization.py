import os
import html
import numpy as np

from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer

from typing import List, Dict, Any

####################################################################################################
# SimpleTokenizer Class:                                                                           #
####################################################################################################

class SimpleTokenizer:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vocab = ['[???]']
        self._train = True
        self._ignored_arguments = []

    char_stems = {
    # a:
        'à': 'a',
        'á': 'a',
        'â': 'a',
        'ã': 'a',
        'ä': 'a',
        'å': 'a',
        'æ': 'a',
        'ā': 'a',
        'ă': 'a',
        'ą': 'a',
        'ấ': 'a',

    # e:
        'è': 'e',
        'é': 'e',
        'ê': 'e',
        'ë': 'e',
        'ē': 'e',
        'ĕ': 'e',
        'ė': 'e',
        'ę': 'e',
        'ě': 'e',
        'ẻ': 'e',

    # i:
        'ì': 'i',
        'í': 'i',
        'î': 'i',
        'ï': 'i',
        'ĩ': 'i',
        'ī': 'i',
        'ĭ': 'i',
        'į': 'i',
        'ı': 'i',

        'ĳ': 'ij',

        'ĵ': 'j',

    # o:
        'ó': 'o',
        'ô': 'o',
        'õ': 'o',
        'ö': 'o',
        'ō': 'o',
        'ŏ': 'o',
        'ő': 'o',
        'œ': 'o',

    # u:    
        'ù': 'u',
        'ú': 'u',
        'û': 'u',
        'ü': 'u',
        'ũ': 'u',
        'ū': 'u',
        'ŭ': 'u',
        'ů': 'u',
        'ű': 'u',
        'ų': 'u',

    # b:
        'ƀ': 'b',

    # c:
        'ç': 'c',
        'ć': 'c',
        'ĉ': 'c',
        'ċ': 'c',
        'č': 'c',

    # d:
        'ď': 'd',
        'đ': 'd',

    # g:
        'ĝ': 'g',
        'ğ': 'g',
        'ġ': 'g',
        'ģ': 'g',

    # h:
        'ĥ': 'h',
        'ħ': 'h',

    # k:
        'ķ': 'k',

    # l:
        'ĺ': 'l',
        'ļ': 'l',
        'ľ': 'l',
        'ŀ': 'l',
        'ł': 'l',

    # n:
        'ñ': 'n',
        'ń': 'n',
        'ņ': 'n',
        'ň': 'n',
        'ŉ': 'n',
        'ŋ': 'n',

    # r:
        'ŕ': 'r',
        'ŗ': 'r',
        'ř': 'r',

    # s:
        'ś': 's',
        'ŝ': 's',
        'ş': 's',
        'š': 's',

    # t:
        'ţ': 't',
        'ť': 't',
        'ŧ': 't',

    # w:
        'ŵ': 'w',

    # y:
        'ý': 'y',
        'ÿ': 'y',
        'ŷ': 'y',

    # z:
        'ź': 'z',
        'ż': 'z',
        'ž': 'z',

    # numbers:
        '0': '', 
        '1': '', 
        '2': '', 
        '3': '', 
        '4': '', 
        '5': '', 
        '6': '', 
        '7': '', 
        '8': '', 
        '9': '',

    # other:
        '„': '', 
        '“': '', 
        '”': '', 
        '«': '', 
        '»': '', 
        '"': '', 
        '‚': '', 
        '‘': '', 
        '’': '', 
        '‹': '', 
        '›': '', 
        '\'': '', 
        '\u200b': '',
        '\u200e': '',
        '®': ''
    }

    separators = [
        ';', ':', ',', '.', '!', '?',
        '(', ')', '[', ']', '{', '}',
        '\t', '\n', '\r',
        '-', '+', '*', '/', '\\', '|', '&',
        '—', '_'
    ]

    @property
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        return len(self._vocab)
    
    @property
    def all_special_tokens(self) -> List[str]:
        """
        `List[str]`: A list of the unique special tokens (`'<unk>'`, `'<cls>'`, ..., etc.).

        Convert tokens of `tokenizers.AddedToken` type to string.
        """
        return self._vocab[self.all_special_ids]

    @property
    def all_special_ids(self) -> List[int]:
        """
        `List[int]`: List the ids of the special tokens(`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.
        """
        return [0]

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
        offset_mapping = []
        input_ids = []

        for argument in kwargs:
            if argument not in self._ignored_arguments:
                print(f'WARNING: ignoring argument "{argument}" in call to SimpleTokenizer.__call__(...)!')
                self._ignored_arguments.append(argument)

        # convert to lowercase:
        text = text.lower()

        # mask xml:
        i = text.find('<')
        j = text.find('>', i+1)+1
        while i >= 0 and j > 0:
            text = text[:i] + ' '*(j-i) + text[j:]

            i = text.find('<', j)
            j = text.find('>', i+1)+1

        # mask html-entities:
        i = text.find('&')
        j = text[:i+10].find(';', i+1)+1
        while i >= 0:
            try: text = text[:i] + html.entities.html5[text[i+1:j]]*(j-i) + text[j:]
            except KeyError: pass

            i = text.find('&', i+1)
            j = text[:i+10].find(';', i+1)+1

        # stem & tokenize:
        start = 0
        for i, c in enumerate(text):
            # skip already processed chars:
            if i < start: continue

            # split at separators:
            if c.isspace() or c in SimpleTokenizer.separators:
                # add word to tokens:
                stem = self._convert_token_to_id(text[start:i])
                if stem >= 0:
                    input_ids.append(stem)
                    offset_mapping.append((start, i))

                # update start:
                for start in range(i+1, min(i+3, len(text))):
                    if text[start].isspace(): continue

                # if no space found:
                start = i + 1

        # add last word:
        stem = self._convert_token_to_id(text[start:])
        if stem >= 0:
            input_ids.append(stem)
            offset_mapping.append((start, len(text)))

        result =  {'input_ids': np.array(input_ids, dtype=int)}

        if return_offsets_mapping:
            result['offset_mapping'] = np.array(offset_mapping, dtype=int)

        return result

    def _convert_token_to_id(self, token:str) -> int:
        stem = ''

        # replace characters with ascii eqivalent:
        for char in html.unescape(token.lower()):
            try: stem += SimpleTokenizer.char_stems[char]
            except KeyError: stem += char

        # drop trailing char 's':
        if len(stem) > 0:
            if stem[-1] == 's':
                stem = stem[:-1]

        # return -1 on empty stems:
        if len(stem) == 0: return -1

        # find index:
        try: return self._vocab.index(stem)
        except ValueError:
            if self._train:
                self._vocab.append(stem)
                return len(self._vocab) - 1
            else: return 0

    def _convert_id_to_token(self, index:int) -> str:
        try: return self._vocab[index]
        except IndexError: return self._vocab[0]

    @staticmethod
    def load(dir:str, **kwargs) -> 'SimpleTokenizer':
        # create WordTokenizer-object:
        tokenizer = SimpleTokenizer(**kwargs)

        # load vocab from file:
        with open(dir + '/vocab.pickle', 'r', encoding='utf-8') as file:
            tokenizer._vocab = file.read().split('\n')
            tokenizer._train = False

        return tokenizer

    def save(self, dir:str) -> None:
        # create directories:
        os.makedirs(dir, exist_ok=True)

        # save vocabulary:
        with open(dir + '/vocab.pickle', 'w', encoding='utf-8') as file:
            file.write('\n'.join(self._vocab))


####################################################################################################
# WordTokenizer Class:                                                                             #
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
    def all_special_tokens(self) -> List[str]:
        """
        `List[str]`: A list of the unique special tokens (`'<unk>'`, `'<cls>'`, ..., etc.).

        Convert tokens of `tokenizers.AddedToken` type to string.
        """
        return self._vocab[self.all_special_ids]
    
    @property
    def all_special_ids(self) -> List[int]:
        """
        `List[int]`: List the ids of the special tokens(`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.
        """
        return [0]

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
        with open(dir + '/vocab.pickle', 'r', encoding='utf-8') as file:
            tokenizer._vocab = file.read().split('\n')
            tokenizer._train = False

        return tokenizer

    def save(self, dir:str) -> None:
        # create directories:
        os.makedirs(dir, exist_ok=True)

        # save vocabulary:
        with open(dir + '/vocab.pickle', 'w', encoding='utf-8') as file:
            file.write('\n'.join(self._vocab))