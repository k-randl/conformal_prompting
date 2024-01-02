import numpy as np

####################################################################################################
# Type hints:                                                                                      #
####################################################################################################

import abc
import numpy.typing as npt
from typing import Iterable, Optional

####################################################################################################
# Naive Models:                                                                                    #
####################################################################################################

class NaiveModel(metaclass=abc.ABCMeta):
    def __init__(self, num_classes:int=2) -> None:
        self._n = num_classes

    @property
    def classes_(self) -> Iterable[int]:
        try: return np.arange(self._n)

        # for backward compatibility:
        except AttributeError:
            self._n = 2
            return np.arange(self._n)

    @abc.abstractmethod
    def predict(self, X:Iterable) -> npt.NDArray:
        raise NotImplementedError()

    @abc.abstractmethod
    def predict_proba(self, X:Iterable) -> npt.NDArray:
        raise NotImplementedError()

    def fit(self, X:Iterable, y:Iterable, w:Optional[Iterable]=None) -> None:
        pass

class SupportModel(NaiveModel):
    def __init__(self, num_classes:int=2) -> None:
        super().__init__(num_classes)

    def predict(self, X:Iterable) -> npt.NDArray:
        return np.ones(len(X), dtype=int) * np.argmax(self._p)
    
    def predict_proba(self, X:Iterable) -> npt.NDArray:
        return np.full((len(X), self._n), self._p, dtype=float)
    
    def fit(self, X:Iterable, y:Iterable, w:Optional[Iterable]=None) -> None:
        m = np.array([y == c for c in self.classes_], dtype=float)

        if w is not None:
            if len(w) == self._n:   m = np.apply_along_axis(lambda col: col*w, 0, m)
            elif len(w) == len(y):  m = np.apply_along_axis(lambda row: row*w, 1, m)
            else: raise ValueError(f'w must either have one entry per class label ({self._n:d}) or per sample ({len(y):d}), but has length {len(w):d}.')

        self._p = np.mean(m, axis=1)

class RandomModel(NaiveModel):
    def __init__(self, num_classes:int=2) -> None:
        super().__init__(num_classes)

    def predict(self, X:Iterable) -> npt.NDArray:
        return np.random.randint(self._n, size=len(X))
    
    def predict_proba(self, X:Iterable) -> npt.NDArray:
        return np.random.random((len(X), self._n))

class DummyModel(NaiveModel):
    def __init__(self, output:int, num_classes:int=2) -> None:
        super().__init__(num_classes)
        self._output = output

    def predict(self, X:Iterable) -> npt.NDArray:
        return np.ones(len(X), dtype=int) * self._output
    
    def predict_proba(self, X:Iterable) -> npt.NDArray:
        p = np.zeros((len(X), self._n), dtype=float)
        p[:, self._output] = 1.
        return p