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
        self._p = np.mean([c == y for c in self.classes_], axis=1)

class RandomModel(NaiveModel):
    def __init__(self, num_classes:int=2) -> None:
        super().__init__(num_classes)

    def predict(self, X:Iterable) -> npt.NDArray:
        return np.random.random(len(X))
    
    def predict_proba(self, X:Iterable) -> npt.NDArray:
        return np.random.random((len(X), self._n))

class DummyModel(NaiveModel):
    def __init__(self, output:int) -> None:
        super().__init__(2)
        self._output = output

    def predict(self, X:Iterable) -> npt.NDArray:
        return np.ones(len(X), dtype=int) * self._output
    
    def predict_proba(self, X:Iterable) -> npt.NDArray:
        p = np.empty((len(X), 2), dtype=float)
        p[:,0] = 1-self._output
        p[:,1] = self._output
        return p