import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse
from typing import Iterable, Tuple, Union, Optional


_T_label_req = Union[int, str]
_T_label_opt = Union[int, str, None]


class ConfusionMatrix:
    def __init__(self, y_true:Iterable[int], y_pred:Iterable[int], classes:Optional[Iterable[str]]=None, num_classes:Optional[int]=None) -> None:
        # Create label-to-index-mapping:
        if classes is not None:
            self._labels = {label:i for i, label in enumerate(classes)}
            num_classes = len(classes)

        elif num_classes is not None:
            self._labels = {f'#{i:d}':i for i in range(num_classes)}

        else: raise AttributeError(f'Either "classes" or "num_classes" have to be different from None!')
        
        # Build matrix:
        self._matrix = np.zeros((num_classes, num_classes), dtype=float)

        for tl, pl in zip(y_true, y_pred):
            self._matrix[tl, pl] += 1

        # Compress sparse matrix:
        self._matrix = sparse.csr_matrix(self._matrix)

        # Save some variables to avoid recalculation:
        self.__last_msk = (None, None)
        self.__last_tp =  (None, None)
        self.__last_fp =  (None, None)
        self.__last_tn =  (None, None)
        self.__last_fn =  (None, None)

    def __getitem__(self, key:Union[Tuple[Union[int, str, slice], Union[int, str, slice]], int, str, slice]):
        if isinstance(key, tuple):
            i, j = key
            if isinstance(i, str): i = self._labels[i]
            if isinstance(j, str): j = self._labels[j]
            key = (i, j)

        elif isinstance(key, str):
            key = self._labels[key]
        
        return self._matrix.__getitem__(key)

    def __sizeof__(self) -> int:
        return self._matrix.__sizeof__()

    def __get_mask(self, label:int):
        # recalculate if needed:
        if self.__last_msk[0] != label:
            self.__last_msk = (label, np.arange(self._matrix.shape[0]) != label)

        return self.__last_msk[1]

    @property
    def shape(self) -> Tuple[int, int]:
        return self._matrix.shape

    @staticmethod
    def from_file(path) -> 'ConfusionMatrix':
        result = ConfusionMatrix([],[],0)
        result.matrix = np.load(path, allow_pickle=True)
        return result

    def save(self, path:str) -> None:
        np.save(path, self._matrix, allow_pickle=True)

    def true_positives(self, label:_T_label_req) -> float:
        # transform label to int:
        if isinstance(label, str):
            label = self._labels[label]

        # recalculate if needed:
        if self.__last_tp[0] != label:
            self.__last_tp = (label, float(self._matrix[label, label]))

        return self.__last_tp[1]

    def true_negatives(self, label:_T_label_req) -> float:
        # transform label to int:
        if isinstance(label, str):
            label = self._labels[label]

        # recalculate if needed:
        if self.__last_tn[0] != label:
            m = self.__get_mask(label)
            self.__last_tn = (label, float(self._matrix[m, m].sum()))

        return self.__last_tn[1]

    def false_positives(self, label:_T_label_req) -> float:
        # transform label to int:
        if isinstance(label, str):
            label = self._labels[label]

        # recalculate if needed:
        if self.__last_fp[0] != label:
            m = self.__get_mask(label)
            self.__last_fp = (label, float(self._matrix[m,label].sum()))

        return self.__last_fp[1]

    def false_negatives(self, label:_T_label_req) -> float:
        # transform label to int:
        if isinstance(label, str):
            label = self._labels[label]

        # recalculate if needed:
        if self.__last_fn[0] != label:
            m = self.__get_mask(label)
            self.__last_fn = (label, float(self._matrix[label,m].sum()))

        return self.__last_fn[1]

    def precision(self, label:_T_label_opt=None) -> float:
        if label is None:
            precision = 0.
            n = self._matrix.shape[0]

            for i in range(n):
                precision += self.precision(i)

            return precision / n

        else:
            tp = self.true_positives(label)
            if tp == 0: return 0.

            fp = self.false_positives(label)
            return tp / (tp + fp)

    def recall(self, label:_T_label_opt=None) -> float:
        if label is None:
            recall = 0.
            n = self._matrix.shape[0]

            for i in range(n):
                recall += self.recall(i)

            return recall / n

        else:
            tp = self.true_positives(label)
            if tp == 0: return 0.

            fn = self.false_negatives(label)
            return tp / (tp + fn)

    def accuracy(self, label:_T_label_opt=None) -> float:
        if label is None:
            accuracy = 0.
            n = self._matrix.shape[0]

            for i in range(n):
                accuracy += self.accuracy(i)

            return accuracy / n

        else:
            t = self.true_positives(label) + self.true_negatives(label)
            if t == 0: return 0.

            f = self.false_positives(label) + self.false_negatives(label)
            return t / (t + f)

    def f1(self, label:_T_label_opt=None) -> float:
        recall    = self.recall(label)
        precision = self.precision(label)

        return (2 * recall * precision) / (recall + precision)
     
    def plot(self, ax:plt.Axes, labels:Union[Iterable[_T_label_req], None]=None) -> None:
        # Get Confusion Matrix and labels:
        rows = None
        if labels is None:
            rows   = list(self._labels.values())
            labels = list(self._labels.keys())

        elif isinstance(labels[0], str):
            rows   = [self._labels[l] for l in labels]

        elif isinstance(labels[0], int):
            rows   = labels
            labels = np.take(list(self._labels.keys()), labels, axis=0)

        else: raise TypeError(labels)

        # Normalize (row-wise) and print:
        confusion = self._matrix[rows,:][:,rows]
        confusion_norm = confusion / np.maximum(1., np.sum(self._matrix[rows], axis=1)).reshape((len(rows), 1))
        ax.imshow(confusion_norm.todense(), interpolation='nearest')

        # Show all ticks and label them with the respective list entries
        if len(labels) <= 20:
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels)
        ax.set_xlabel('Predicted Class')
        if len(labels) <= 20:
            ax.set_yticks(np.arange(len(labels)))
            ax.set_yticklabels(labels)
        ax.set_ylabel('True Class')

        # Rotate the tick labels and set their alignment.
        ax.xaxis.set_label_position('top')
        ax.xaxis.set_ticks_position('top')
        if len(labels) <= 5:
            plt.setp(ax.get_yticklabels(), rotation=90, ha="center", rotation_mode="anchor")
            plt.tick_params(left=False, top=False)
        else:
            plt.setp(ax.get_xticklabels(), rotation=-45, ha="right", rotation_mode="anchor")
            plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        if len(labels) <= 20:
            for i in range(len(labels)):
                for j in range(len(labels)):
                    if confusion[i, j] > 0.01:
                        ax.text(j, i, f"{confusion[i, j]:.0f}", ha="center", va="center", color="w")

