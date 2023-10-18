import abc
import numpy as np
import torch.nn as nn

####################################################################################################
# Type hints:                                                                                      #
####################################################################################################

import numpy.typing as npt
from typing import Callable, Iterable, Union, Literal, Dict, Any
from resources.data_io import T_data

T_norm = Union[
    Callable[[npt.NDArray], npt.NDArray],
    Literal["max"],
    Literal["sum"],
    Literal["min-max"],
    Literal["softmax"],
    None
]

####################################################################################################
# Normalize Functions:                                                                             #
####################################################################################################

def _norm_identity(x:npt.NDArray) -> npt.NDArray:   return x
def _norm_max(x:npt.NDArray) -> npt.NDArray:        return x / max(x.max(), 1e-6)
def _norm_sum(x:npt.NDArray) -> npt.NDArray:        return x / max(x.sum(), 1e-6)
def _norm_min_max(x:npt.NDArray) -> npt.NDArray:    return _norm_max(x - x.min())
def _norm_softmax(x:npt.NDArray) -> npt.NDArray:    return _norm_sum(np.exp(x))

####################################################################################################
# getNormFcn:                                                                                      #
####################################################################################################

def getNormFcn(normalize_fcn:T_norm) -> Callable[[npt.NDArray], npt.NDArray]:
    '''Returns a normalization function.'''
    if normalize_fcn is None:               return _norm_identity
    elif isinstance(normalize_fcn, str):
        if   normalize_fcn == "softmax":    return _norm_softmax
        elif normalize_fcn == "min-max":    return _norm_min_max
        elif normalize_fcn == "max":        return _norm_max
        elif normalize_fcn == "sum":        return _norm_sum
        else: raise ValueError(normalize_fcn)

    else:                                   return normalize_fcn

####################################################################################################
# Evaluator Class:                                                                                 #
####################################################################################################

class Evaluator(metaclass=abc.ABCMeta):
    def __init__(self, normalize_fcn:T_norm=None, num_labels:int=0) -> None:
        self._model         = None
        self._num_labels    = num_labels
        self._normalize_fcn = getNormFcn(normalize_fcn)

    @property
    def last_spans(self) -> Union[Iterable[npt.NDArray], npt.NDArray, None]:
        '''`Iterable[NDArray] | NDArray | None]`: the spans of the last evaluated dataset as a binary mask (per token).'''
        if hasattr(self, '_last_spans'):
            return self._last_spans
        else: return None

    @property
    def last_texts(self) -> Union[Iterable[str], None]:
        '''`Iterable[str] | None]`: the texts of the last evaluated dataset.'''
        if hasattr(self, '_last_texts'):
            return self._last_texts
        else: return None

    @property
    def num_labels(self) -> int:
        '''`int`: the number of unique labels predicted by the model.'''
        self._num_labels

    @abc.abstractstaticmethod
    def load(dir:str, normalize_fcn:T_norm=None, **kwargs) -> 'Evaluator':
        '''Loads a model from disk.

            dir:           Path to the directory that contains the model (e.g.: .\models)

            normalize_fcn: Normalization function used on the predictions [`"max"` | `"sum"` | `"min-max"` | `"softmax"` | `None`]

            
            returns:       `Evaluator` of the model
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def save(self, dir:str, **kwargs) -> None:
        '''Saves a model to disk.

            dir:           Path to the directory that contains the model (e.g.: .\models)
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, data:T_data, output_spans:bool=False, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError()


class EvaluatorMirror(Evaluator):
    def __init__(self, obj:Union[Evaluator, None]=None) -> None:
        super().__init__(obj._normalize_fcn, obj._num_labels)
        self._model = obj._model
        self._base = obj

    @property
    def last_spans(self) -> Union[Iterable[npt.NDArray], npt.NDArray, None]:
        '''`Iterable[NDArray] | NDArray | None]`: the spans of the last evaluated dataset as a binary mask (per token).'''
        if self._base is not None:
            return self._base.last_spans
        else: return None

    @property
    def last_texts(self) -> Union[Iterable[str], None]:
        '''`Iterable[str] | None]`: the texts of the last evaluated dataset.'''
        if self._base is not None:
            return self._base.last_texts
        else: return None

    @staticmethod
    def load(dir:str, normalize_fcn:T_norm=None, **kwargs) -> 'EvaluatorMirror':
        '''Loads a model from disk.

            dir:           Path to the directory that contains the model (e.g.: .\models)

            normalize_fcn: Normalization function used on the predictions [`"max"` | `"sum"` | `"min-max"` | `"softmax"` | `None`]

            
            returns:       `Evaluator` of the model
        '''
        return NotImplementedError('The generic `EvaluatorMirror` class does not support the load method.')

    def save(self, dir:str, **kwargs) -> None:
        '''Saves a model to disk.

            dir:           Path to the directory that contains the model (e.g.: .\models)
        '''
        if self._base is not None:
            return self._base.save(dir, **kwargs)
        else: raise RuntimeError(f'Instance of `{self.__name__}` has not been instatiated with an object. Save function not available.')

    def predict(self, data:T_data, output_spans:bool=False, **kwargs) -> Dict[str, Any]:
        if self._base is not None:
            return self._base.predict(data, output_spans, **kwargs)
        else: raise RuntimeError(f'Instance of `{self.__name__}` has not been instatiated with an object. Predict function not available.')


class EvaluatorThreshold(EvaluatorMirror):
    def predict(self, epsilon:float, data:Union[T_data,None]=None, y_pred:Union[npt.NDArray,None]=None, **kwargs) -> Dict[str, any]:
        results = {}

        assert not(((data is None) or (self._model is None)) and (y_pred is None))
        if not (data is None):
            results = super().predict(data, **kwargs)
            y_pred = results['predictions']

        preds = y_pred.argsort(axis=1)[:,::-1]
        probs = np.take_along_axis(y_pred, preds, axis=1)

        y_thld = []
        for i in range(y_pred.shape[0]):
            prediction_set = []

            for y, p in zip(preds[i], probs[i]):
                prediction_set.append((y, p))
                if p < (1. - epsilon): break

            y_thld.append(np.array(prediction_set, dtype=np.dtype([('i', 'u4'), ('p', 'f4')])))

        results['predictions'] = y_thld
        return results


class EvaluatorMaxK(EvaluatorMirror):
    def predict(self, k:int, data:Union[T_data,None]=None, y_pred:Union[npt.NDArray,None]=None, **kwargs) -> Dict[str, any]:
        results = {}

        assert not(((data is None) or (self._model is None)) and (y_pred is None))
        if not (data is None):
            results = super().predict(data, **kwargs)
            y_pred = results['predictions']

        preds = y_pred.argsort(axis=1)[:,::-1]
        probs = np.take_along_axis(y_pred, preds, axis=1)

        y_max_k = []
        for i in range(y_pred.shape[0]):
            y_max_k.append(
                np.array(
                    [(preds[i,j], probs[i,j]) for j in range(k)],
                    dtype=np.dtype([('i', 'u4'), ('p', 'f4')])
                )
            )

        results['predictions'] = y_max_k
        return results


class EvaluatorConformal(EvaluatorMirror, metaclass=abc.ABCMeta):
    def _get_predictions(self, data:Union[T_data,None], y_pred:Union[npt.NDArray,None], y_true:Union[npt.NDArray,None]=None, **kwargs) -> Dict[str, any]:
        assert not(((data is None) or (self._model is None)) and (y_pred is None))
        if data is None: return {'labels':y_true, 'predictions':y_pred}
        else:            return super().predict(data, **kwargs)

    @abc.abstractmethod
    def quantile(self, epsilon:float) -> Union[Iterable[float], float]: pass

    @abc.abstractmethod
    def calibrate(self, data:Union[T_data,None]=None, y_pred:Union[npt.NDArray,None]=None, y_true:Union[npt.NDArray,None]=None) -> None: pass

    @abc.abstractmethod
    def predict(self, epsilon:float, data:Union[T_data,None]=None, y_pred:Union[npt.NDArray,None]=None, loss_fcn:Union[nn.Module, None]=None, output_attentions:bool=False, output_hidden_states:bool=False) -> Dict[str, any]: pass


class EvaluatorConformalAPS(EvaluatorConformal):
    def quantile(self, epsilon:float) -> float:
        return np.quantile(
            self.alphas,
            np.ceil((len(self.alphas)+1)*(1-epsilon))/len(self.alphas),
            interpolation='higher'
        )

    def calibrate(self, data:Union[T_data,None]=None, y_pred:Union[npt.NDArray,None]=None, y_true:Union[npt.NDArray,None]=None) -> None:
        results = self._get_predictions(data, y_pred, y_true)
        y_true = results['labels']
        y_pred = results['predictions']
        assert not (y_true is None)
        assert len(y_true) == len(y_pred)

        preds = y_pred.argsort(axis=1)[:,::-1]
        probs = np.cumsum(np.take_along_axis(y_pred, preds, axis=1), axis=1)
        idxs  = np.argwhere(np.take_along_axis(y_true, preds, axis=1))

        self.alphas = probs[idxs]

    def predict(self, epsilon:float, data:Union[T_data,None]=None, y_pred:Union[npt.NDArray,None]=None, **kwargs) -> Dict[str, any]: 
        results = self._get_predictions(data, y_pred, **kwargs)
        y_pred  = results['predictions']

        preds = y_pred.argsort(axis=1)[:,::-1]
        probs = np.take_along_axis(y_pred, preds, axis=1)

        q = self.quantile(epsilon)

        y_conf = []
        for i in range(y_pred.shape[0]):
            prediction_set = []
            total_p = 0.

            for y, p in zip(preds[i], probs[i]):
                prediction_set.append((y, p))
                total_p += p
                if total_p >= q: break

            y_conf.append(np.array(prediction_set, dtype=np.dtype([('i', 'u4'), ('p', 'f4')])))

        results['predictions'] = y_conf
        return results


class EvaluatorConformalSimple(EvaluatorConformal):
    def quantile(self, epsilon:float) -> float:
        return np.quantile(
            self.alphas,
            np.ceil((len(self.alphas)+1)*(1-epsilon))/len(self.alphas),
            interpolation='higher'
        )

    def calibrate(self, data:Union[T_data,None]=None, y_pred:Union[npt.NDArray,None]=None, y_true:Union[npt.NDArray,None]=None) -> None:
        results = self._get_predictions(data, y_pred, y_true)
        y_true = results['labels']
        y_pred = results['predictions']
        assert not (y_true is None)
        assert len(y_true) == len(y_pred)

        self.alphas = 1. - y_pred[y_true.astype(bool)]

    def predict(self, epsilon:float, data:Union[T_data,None]=None, y_pred:Union[npt.NDArray,None]=None, **kwargs) -> Dict[str, any]: 
        results = self._get_predictions(data, y_pred, **kwargs)
        y_pred  = results['predictions']

        q = self.quantile(epsilon)

        y_conf = []
        for i in range(y_pred.shape[0]):
            preds = np.argwhere(y_pred[i] >= (1.-q))[:,0]
            preds = preds[np.argsort(y_pred[i, preds])][::-1]

            y_conf.append(
                np.array(
                    [(y, y_pred[i, y]) for y in preds],
                    dtype=np.dtype([('i', 'u4'), ('p', 'f4')])
                )
            )

        results['predictions'] = y_conf
        return results

####################################################################################################
# Trainer Class:                                                                                   #
####################################################################################################

class Trainer(Evaluator, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self,
            model:str,
            data_train:T_data,
            data_valid:T_data,
            bias:float=0.5,
            modelargs:Dict[str, Iterable[Any]]={}
        ) -> None:
        raise NotImplementedError()

    @property
    def train_history(self) -> Dict[str, float]:
        '''`Dict[str, float]`: the scores of the last trained classifiers.'''
        if hasattr(self, '_train_history'):
            return self._train_history
        else: return {}