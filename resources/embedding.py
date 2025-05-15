import os
import abc
import pickle
import numpy as np

from tqdm.autonotebook import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score
from resources.tokenization import WordTokenizer
from resources.multiprocessing import ArgumentQueue

####################################################################################################
# Type hints:                                                                                      #
####################################################################################################

import numpy.typing as npt
from typing import Iterable, Tuple, Union, Optional, Literal
from resources.data_io import T_data

####################################################################################################
# Plotting:                                                                                        #
####################################################################################################

try: import matplotlib.pyplot as plt
except ModuleNotFoundError: plt=None

####################################################################################################
# Thread Functions:                                                                                #
####################################################################################################

#def _calculate_idf(id, input_ids):
#    # count words:
#    row = np.empty(len(input_ids), dtype=float)
#    for i, doc in enumerate(input_ids):
#        row[i] = (doc == id).sum()
#    return id, -np.log((row > 0).mean(dtype=float))

def _calculate_idf(id, input_ids):
    count = 0

    # count documents:
    for doc in input_ids:
        count += int(id in doc)
    
    if count == 0:  return id, 0.
    else:           return id, np.log(float(len(input_ids)) / float(count))

####################################################################################################
# Embedding Class:                                                                                 #
####################################################################################################

class Embedding(metaclass=abc.ABCMeta):
    def __init__(self, tokenizer:WordTokenizer) -> None:
        self._tokenizer = tokenizer
        self._pca       = None

    def __call__(self, data:Union[Iterable[str], Iterable[Iterable[int]]], compress:bool=True) -> npt.NDArray:
        '''Encodes texts.

            data:       An iterable of `n` tokenized or raw texts.
            compress:   Use PCA to compress the vector if available.

            
            returns:    The encoded texts as an numpy-array with `n` rows.
        '''
        encodings = np.zeros((len(data), self.num_tokens), dtype=float)

        for i, value in enumerate(data):
            # tokenize if necessary:
            if isinstance(value, str):
                value = self._tokenizer(value, return_offsets_mapping=False)['input_ids']

            # create encoding:
            encodings[i] = self._encode_single(value)

        # compress encodings:
        if compress and (self._pca is not None):
            encodings = self._pca.transform(encodings)

        return encodings
    
    @abc.abstractmethod
    def _encode_single(self, input_ids:Iterable[int]) -> Iterable[float]:
        '''Encodes a single text.

            input_ids:  A tokenized text.

            
            returns:    The encoded text.
        '''
        raise NotImplementedError()

    @property
    def tokenizer(self) -> WordTokenizer:
        return self._tokenizer

    @property
    def num_tokens(self) -> int:
        return self._tokenizer.vocab_size

    def fit_pca(self, data:Union[Iterable[str], Iterable[Iterable[int]]], n_components:Optional[Union[int, float, Literal['mle']]]=None, verbose:bool=False):
        if verbose: print('\nReducing dimensionality...')
        encoded_data = self(data)
        self._pca = PCA(n_components=n_components)
        reduced_data = self._pca.fit_transform(encoded_data)
        if verbose: print(f'Done. Reduced dimensionality from {encoded_data.shape} to {reduced_data.shape}.')

    def cosine_similarity(self, data:Union[Iterable[str], Iterable[Iterable[int]]]) -> npt.NDArray:
        return 1. - pairwise_distances(self(data, compress=False), metric='cosine')
    
    def silhouette_score(self, x:Union[Iterable[str], Iterable[Iterable[int]]], y:Union[Iterable[str], Iterable[int]]):
        return silhouette_score(self(x, compress=False), y, metric='cosine')
    
    def scatter(self, x:Union[Iterable[str], Iterable[Iterable[int]]], y:Union[Iterable[str], Iterable[int]], **kwargs) -> npt.NDArray:
        # compute pairwise distances:
        distance = pairwise_distances(self(x, compress=False), metric='cosine')
        
        if plt is None: print("!!!WARNING!!! Module 'matplotlib' could not be found")
        else:
            # convert labels to integer if necessary:
            if not isinstance(y[0], int):
                y2i = {label:i for i,label in enumerate(np.unique(y))}
                y = np.apply_along_axis(lambda label: y2i[label[0]], -1, np.reshape(y, y.shape + (1,)))

            # compress to 2d using pca:
            pca = PCA(2)
            distance_2d = pca.fit_transform(distance)
            
            # plot data:
            plt.scatter(distance_2d[:,0], distance_2d[:,1], c=y, **kwargs)

        return distance

    @abc.abstractstaticmethod
    def load(dir:str, **kwargs) -> 'Embedding':
        '''Loads an `Embedding` object from disk.

            dir:       Path to the directory that contains the `Embedding` (e.g.: .\models)

            
            returns:   An `Embedding` object.
        '''
        files = os.listdir(dir)
        if 'bow.pickle' in files: return EmbeddingBOW.load(dir, **kwargs)
        elif 'tf-idf.pickle' in files: return EmbeddingTfIdf.load(dir, **kwargs)
        else: raise NotImplementedError()

    @abc.abstractmethod
    def save(self, dir:str, **kwargs) -> None:
        '''Saves an `Embedding` object to disk.

            dir:       Path to the directory that contains the `Embedding` (e.g.: .\models)
        '''
        raise NotImplementedError()
    
####################################################################################################
# Bag-of-Words Embedding Class:                                                                    #
####################################################################################################

class EmbeddingBOW(Embedding):
    def __init__(self, tokenizer:WordTokenizer) -> None:
        super().__init__(tokenizer)

    def _encode_single(self, input_ids: Iterable[int]) -> Iterable[float]:
        '''Encodes a single text.

            input_ids:  A tokenized text.

            
            returns:    The encoded text.
        '''
        encoded_ids = np.zeros(self.num_tokens, dtype=float)

        # create bag-of-words-vector:
        for t in input_ids:
            if t < self.num_tokens:
                encoded_ids[t] += 1

        # normalize bag-of-words-vector:
        np.seterr(divide='ignore', invalid='ignore')
        encoded_ids = np.nan_to_num(encoded_ids / np.linalg.norm(encoded_ids))
        np.seterr(divide='warn', invalid='warn')

        return encoded_ids

    @staticmethod
    def load(dir:str, **kwargs) -> 'EmbeddingBOW':
        '''Loads an `EmbeddingBOW` object from disk.

            dir:       Path to the directory that contains the `EmbeddingBOW` (e.g.: .\models)

            
            returns:   An `EmbeddingBOW` object.
        '''
        # load tokeinzer instance:
        tokenizer = WordTokenizer.load(dir, **kwargs)

        # create model instance:
        bow = EmbeddingBOW(tokenizer)

        try:
            # load data from file:
            with open(dir + '/bow.pickle', 'rb') as file:
                data = pickle.load(file)

            # update model parameters:
            bow._pca = data['pca']
        except: pass

        return bow

    def save(self, dir:str, **kwargs) -> None:
        '''Saves an `EmbeddingBOW` object to disk.

            dir:       Path to the directory that contains the `EmbeddingBOW` (e.g.: .\models)
        '''
        # create directories:
        os.makedirs(dir, exist_ok=True)

        # save model:
        with open(dir + '/bow.pickle', 'wb') as file:
            pickle.dump({
                'pca': self._pca
            }, file)

        # save tokenizer:
        self._tokenizer.save(dir)

####################################################################################################
# TF-IDF Embedding Class:                                                                          #
####################################################################################################

class EmbeddingTfIdf(Embedding):
    def __init__(self, data:Union[Iterable[str], Iterable[Iterable[int]]], tokenizer:Optional[WordTokenizer]=None, num_workers:int=5) -> None:
        super().__init__(tokenizer)

        # stop on empty data:
        if len(data) == 0: return

        # create and train new tokenizer if necessary:
        if tokenizer is None:
            if not isinstance(data[0], str):
                raise TypeError('Parameter "data" must be an Iterable of strings if no tokenizer is passed.')

            self._tokenizer = WordTokenizer()
            self._tokenizer.train = True  

        # unpack data:
        if isinstance(data[0], str):
            data = [self._tokenizer(input_ids, return_offsets_mapping=False)['input_ids'] for input_ids in tqdm(data, desc='Tokenizing texts')]

        # end tokenizer training:
        if tokenizer is None: self._tokenizer.train = False

        # calculate inverse document frequency per sample:
        self._idf = np.zeros(self.num_tokens, dtype=float)
        queue = ArgumentQueue(_calculate_idf, [(i,) for i in range(self.num_tokens)], desc='Calculating inverse document frequency')
        for i, v in queue(n_workers=num_workers, input_ids=data, timeout=1.): self._idf[i] = v
    
    def _encode_single(self, input_ids:Iterable[int]) -> Iterable[float]:
        '''Encodes a single text.

            input_ids:  A tokenized text.

            
            returns:    The encoded text.
        '''
        encoded_ids = np.zeros(self.num_tokens, dtype=float)

        # create tf-idf-vector:
        for t in input_ids:
            if t < self.num_tokens:
                encoded_ids[t] += self._idf[t] / len(input_ids)

        # normalize tf-idf-vector:
        np.seterr(divide='ignore', invalid='ignore')
        encoded_ids = np.nan_to_num(encoded_ids / np.linalg.norm(encoded_ids))
        np.seterr(divide='warn', invalid='warn')

        return encoded_ids

    @property
    def idf(self) -> Iterable[Tuple[str, float]]:
        return list(zip(self._tokenizer._vocab, self._idf))

    @staticmethod
    def load(dir:str, **kwargs) -> 'EmbeddingTfIdf':
        '''Loads an `EmbeddingTfIdf` object from disk.

            dir:       Path to the directory that contains the `EmbeddingTfIdf` (e.g.: .\models)

            
            returns:   An `EmbeddingTfIdf` object.
        '''
        # load tokeinzer instance:
        tokenizer = WordTokenizer.load(dir, **kwargs)

        # create embedding instance:
        embedding = EmbeddingTfIdf([], tokenizer)

        # load data from file:
        with open(dir + '/tf-idf.pickle', 'rb') as file:
            data = pickle.load(file)

        # update embedding parameters:
        embedding._idf = data['idf']
        embedding._pca = data['pca']

        return embedding

    def save(self, dir:str, **kwargs) -> None:
        '''Saves an `EmbeddingTfIdf` object to disk.

            dir:       Path to the directory that contains the `EmbeddingTfIdf` (e.g.: .\models)
        '''
        # create directories:
        os.makedirs(dir, exist_ok=True)

        # save model:
        with open(dir + '/tf-idf.pickle', 'wb') as file:
            pickle.dump({
                'idf': self._idf,
                'pca': self._pca
            }, file)

        # save tokenizer:
        self._tokenizer.save(dir)