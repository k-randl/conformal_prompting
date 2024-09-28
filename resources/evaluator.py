import abc
import pickle
import numpy as np

from sklearn.base import ClassifierMixin

####################################################################################################
# Type hints:                                                                                      #
####################################################################################################

import numpy.typing as npt
from typing import Callable, Iterable, Union, Optional, Literal, List, Dict, Any
from resources.data_io import T_data

T_norm = Union[
    Callable[[npt.NDArray], npt.NDArray],
    Literal["max"],
    Literal["sum"],
    Literal["min-max"],
    Literal["softmax"],
    Literal["argmax"],
    None
]

####################################################################################################
# Normalize Functions:                                                                             #
####################################################################################################

def _norm_identity(x:npt.NDArray) -> npt.NDArray:   return x
def _norm_argmax(x:npt.NDArray) -> npt.NDArray:     return np.array([np.argmax(x)])
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
        if   normalize_fcn == "argmax":     return _norm_argmax
        elif normalize_fcn == "softmax":    return _norm_softmax
        elif normalize_fcn == "min-max":    return _norm_min_max
        elif normalize_fcn == "max":        return _norm_max
        elif normalize_fcn == "sum":        return _norm_sum
        else: raise ValueError(normalize_fcn)

    else:                                   return normalize_fcn

####################################################################################################
# Evaluator Class:                                                                                 #
####################################################################################################

class Evaluator(metaclass=abc.ABCMeta):
    def __init__(self, model:ClassifierMixin=None, labels:List[str]=[], normalize_fcn:T_norm=None) -> None:
        self._model         = model
        self._labels        = labels
        self._normalize_fcn = getNormFcn(normalize_fcn)

    @property
    def last_spans(self) -> Union[Iterable[npt.NDArray], npt.NDArray, None]:
        '''`Iterable[NDArray] | NDArray | None`: the spans of the last evaluated dataset as a binary mask (per token).'''
        if hasattr(self, '_last_spans'):
            return self._last_spans
        else: return None

    @property
    def last_texts(self) -> Union[Iterable[str], None]:
        '''`Iterable[str] | None`: the texts of the last evaluated dataset.'''
        if hasattr(self, '_last_texts'):
            return self._last_texts
        else: return None

    @property
    def num_labels(self) -> int:
        '''`int`: the number of unique labels predicted by the model.'''
        return len(self._labels)

    @abc.abstractproperty
    def tokenizer(self) -> Callable[[str], Iterable[int]]:
        '''A callable that tokenizes strings to be used by the model.'''
        raise NotImplementedError()

    @staticmethod
    def load(dir:str, normalize_fcn:T_norm=None, **kwargs) -> 'Evaluator':
        '''Loads a model from disk.

            dir:           Path to the directory that contains the model (e.g.: .\models)

            normalize_fcn: Normalization function used on the predictions [`"max"` | `"sum"` | `"min-max"` | `"softmax"` | `None`]

            
            returns:       `Evaluator` of the model
        '''
        # load data from file:
        with open(dir + '/model.pickle', 'rb') as file:
            data = pickle.load(file)

        # call type-specific load function:
        return data['type'].load(dir=dir, normalize_fcn=normalize_fcn, **kwargs)

    @abc.abstractmethod
    def save(self, dir:str, **kwargs) -> None:
        '''Saves a model to disk.

            dir:           Path to the directory that contains the model (e.g.: .\models)
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, data:T_data, **kwargs) -> Dict[str, Any]:
        '''Predicts the output of a model.

            data:          Texts to be assessed by the model.


            returns:       `Dict[str, Any]` of model outputs.
        '''
        raise NotImplementedError()

    def id2label(self, id:Union[int,List[int]]) -> Union[str,List[str]]:
        '''Returns the string label corresponding to an integer model output.

            id:            Integer version of the label.


            returns:       String version of the label.
        '''
        return self._labels[id]

class EvaluatorMirror(Evaluator):
    def __init__(self, obj:Optional[Evaluator]=None, labels:List[str]=[], normalize_fcn:T_norm=None) -> None:
        super().__init__(
            model         = None if obj is None else obj._model,
            labels        = labels if obj is None else obj._labels,
            normalize_fcn = normalize_fcn if obj is None else obj._normalize_fcn
        )
        self._base = obj

    @property
    def last_spans(self) -> Union[Iterable[npt.NDArray], npt.NDArray, None]:
        '''`Iterable[NDArray] | NDArray | None`: the spans of the last evaluated dataset as a binary mask (per token).'''
        if self._base is not None:
            return self._base.last_spans
        else: return None

    @property
    def last_texts(self) -> Union[Iterable[str], None]:
        '''`Iterable[str] | None]`: the texts of the last evaluated dataset.'''
        if self._base is not None:
            return self._base.last_texts
        else: return None

    @property
    def tokenizer(self) -> Callable[[str], Iterable[int]]:
        '''A callable that tokenizes strings to be used by the model.'''
        if self._base is not None:
            return self._base.tokenizer
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

    def predict(self, data:T_data, **kwargs) -> Dict[str, Any]:
        if self._base is not None:
            return self._base.predict(data, **kwargs)
        else: raise RuntimeError(f'Instance of `{self.__name__}` has not been instantiated with an object. Predict function not available.')


class EvaluatorThreshold(EvaluatorMirror):
    def predict(self, alpha:float, data:Optional[T_data]=None, y_pred:Optional[npt.NDArray]=None, **kwargs) -> Dict[str, any]:
        results = {}

        assert not(((data is None) or (self._model is None)) and (y_pred is None))
        if data is not None:
            results = super().predict(data, output_structured=False, **kwargs)
            y_pred = results['probabilities']

        else: y_pred = np.apply_along_axis(self._normalize_fcn, -1, y_pred)

        preds = y_pred.argsort(axis=1)[:,::-1]
        probs = np.take_along_axis(y_pred, preds, axis=1)

        y_thld = []
        for i in range(y_pred.shape[0]):
            prediction_set = []

            for y, p in zip(preds[i], probs[i]):
                prediction_set.append((self.id2label(y), p))
                if p < (1. - alpha): break

            y_thld.append(np.array(prediction_set, dtype=np.dtype([('y', 'object'), ('p', 'f4')])))

        results['predictions'] = y_thld
        return results


class EvaluatorMaxK(EvaluatorMirror):
    def predict(self, k:int, data:Optional[T_data]=None, y_pred:Optional[npt.NDArray]=None, **kwargs) -> Dict[str, any]:
        results = {}

        assert not(((data is None) or (self._model is None)) and (y_pred is None))
        if data is not None:
            results = super().predict(data, output_structured=False, **kwargs)
            y_pred = results['probabilities']

        else: y_pred = np.apply_along_axis(self._normalize_fcn, -1, y_pred)

        preds = y_pred.argsort(axis=1)[:,::-1]
        probs = np.take_along_axis(y_pred, preds, axis=1)

        y_max_k = []
        for i in range(y_pred.shape[0]):
            y_max_k.append(
                np.array(
                    [(self.id2label(preds[i,j]), probs[i,j]) for j in range(k)],
                    dtype=np.dtype([('y', 'object'), ('p', 'f4')])
                )
            )

        results['predictions'] = y_max_k
        return results


class EvaluatorConformal(EvaluatorMirror, metaclass=abc.ABCMeta):
    def _get_predictions(self, data:Optional[T_data], y_pred:Optional[npt.NDArray], y_true:Optional[npt.NDArray]=None, **kwargs) -> Dict[str, any]:
        assert not(((data is None) or (self._model is None)) and (y_pred is None))
        if data is None: return {'labels':y_true, 'probabilities':np.apply_along_axis(self._normalize_fcn, -1, y_pred)}
        else:            return super().predict(data, output_probabilities=True, output_structured=False, **kwargs)

    def quantile(self, alpha:float) -> float:
        # get number of calibration samples:
        n = self.cal_scores.shape[0]

        # calculate per class quantile:
        q = np.nanquantile(
            self.cal_scores,
            np.ceil((n+1.)*(1.-alpha))/n,
            method='higher',
            axis=0
        )

        # set quantile for classes not covered in the calibration set to 1:
        q[np.isnan(q)] = 1.

        return q

    @abc.abstractmethod
    def calibrate(self, data:Optional[T_data]=None, y_pred:Optional[npt.NDArray]=None, y_true:Optional[npt.NDArray]=None) -> None: pass

    @abc.abstractmethod
    def predict(self, alpha:float, data:Optional[T_data]=None, y_pred:Optional[npt.NDArray]=None, min_k:Optional[int]=None, **kwargs) -> Dict[str, any]: pass


class EvaluatorConformalAPS(EvaluatorConformal):
    def calibrate(self, data:Optional[T_data]=None, y_pred:Optional[npt.NDArray]=None, y_true:Optional[npt.NDArray]=None) -> None:
        results = self._get_predictions(data, y_pred, y_true)
        y_true = results['labels']
        y_pred = results['probabilities']
        assert not (y_true is None)
        assert len(y_true) == len(y_pred)

        preds = y_pred.argsort(axis=1)[:,::-1]
        probs = np.cumsum(np.take_along_axis(y_pred, preds, axis=1), axis=1)

        self.cal_scores = np.take_along_axis(probs, preds, axis=1)[np.apply_along_axis(lambda y: y == self._labels, 1, y_true[:,np.newaxis])]

    def predict(self, alpha:float, data:Optional[T_data]=None, y_pred:Optional[npt.NDArray]=None, min_k:Optional[int]=None, **kwargs) -> Dict[str, any]: 
        results = self._get_predictions(data, y_pred, **kwargs)
        y_pred  = results['probabilities']

        preds = y_pred.argsort(axis=1)[:,::-1]
        probs = np.cumsum(np.take_along_axis(y_pred, preds, axis=1), axis=1)

        q = self.quantile(alpha)

        y_conf = []
        for i in range(y_pred.shape[0]):
            ps = preds[i, probs[i] <= q]
            ps = [(self.id2label(y), y_pred[i, y]) for y in ps]

            if min_k is not None:
                for y in np.argsort(y_pred[i])[::-1][len(ps):min_k]:
                    ps.append((self.id2label(y), y_pred[i, y]))

            ps.sort(key=lambda e: e[1], reverse=True)

            y_conf.append(
                np.array(ps, dtype=np.dtype([('y', 'object'), ('p', 'f4')]))
            )

        results['predictions'] = y_conf
        return results


class EvaluatorConformalSimple(EvaluatorConformal):
    def calibrate(self, data:Optional[T_data]=None, y_pred:Optional[npt.NDArray]=None, y_true:Optional[npt.NDArray]=None) -> None:
        results = self._get_predictions(data, y_pred, y_true)
        y_true = results['labels']
        y_pred = results['probabilities']
        assert not (y_true is None)
        assert len(y_true) == len(y_pred)

        # calculate error towards true class:
        m_true = np.apply_along_axis(lambda y: y == self._labels, 1, y_true[:,np.newaxis])
        self.cal_scores = np.full(y_pred.shape, np.nan)
        self.cal_scores[m_true] = 1. - y_pred[m_true]

    def predict(self, alpha:float, data:Optional[T_data]=None, y_pred:Optional[npt.NDArray]=None, min_k:Optional[int]=None, **kwargs) -> Dict[str, any]: 
        results = self._get_predictions(data, y_pred, **kwargs)
        y_pred  = results['probabilities']

        q = self.quantile(alpha)

        y_conf = []
        for i in range(y_pred.shape[0]):
            ps = np.argwhere((1. - y_pred[i]) <= q)[:,0]
            ps = [(self.id2label(y), y_pred[i, y]) for y in ps]

            if min_k is not None:
                for y in np.argsort(y_pred[i])[::-1][len(ps):min_k]:
                    ps.append((self.id2label(y), y_pred[i, y]))

            ps.sort(key=lambda e: e[1], reverse=True)

            y_conf.append(
                np.array(ps, dtype=np.dtype([('y', 'object'), ('p', 'f4')]))
            )

        results['predictions'] = y_conf
        return results

####################################################################################################
# Trainer Class:                                                                                   #
####################################################################################################

class Trainer(Evaluator, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self,
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