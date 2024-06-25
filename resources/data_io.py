import pickle
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from .spans import SpanCollection

####################################################################################################
# Type hints:                                                                                      #
####################################################################################################

from typing import Optional, Tuple, Iterable, Union, Callable

T_data = Union[
    Tuple[Dataset, Iterable[str]],
    Dataset
]

####################################################################################################
# Dataset Class:                                                                                   #
####################################################################################################

class ClassificationDataset(Dataset):
    def __init__(self,
            data:pd.DataFrame,
            text_column:str,
            tokenizer:PreTrainedTokenizerBase,
            label_column:Optional[str]=None,
            label2int:Optional[Callable[[str],int]]=None,
            add_spans:bool=False,
            add_texts:bool=False,
            truncate:bool=True,
            stride:int=1,
            filter_samples:bool=False
        ) -> None:
        super().__init__()

        self.X, self.y = [], []
        self.spans = [] if add_spans else None
        self.data_size = 0

        # extract labels and texts:
        for index, text in enumerate(data[text_column].values):
            label = 0 if label_column is None else data.iloc[index,label_column]
            spans = SpanCollection([]) if label_column is None else data.iloc[index,label_column.split('-')[0] + '-' + text_column]

            separators = [0, len(text)]

            if not truncate:
                # split text into sentences:
                # TODO: Use sent_tokenize?
                for char in ['.', '!', '?', '\n', '\r']:
                    i = text.find(char)
                    while i >= 0:
                        separators.append(i)
                        i = text.find(char, i+1)

                # remove doubles:
                separators = list(set(separators))

                # sort:
                separators.sort()

            # convert label to int:
            if label2int is not None:
                label = label2int(label)

            offset = min(stride, len(separators)-1)
            for start, stop in zip(separators[:-offset],separators[offset:]):
                # extract corresponding spans:
                s = spans.limit(start, stop) << start

                # if filtering activated -> dismiss sequences without spans:
                if filter_samples and len(s) == 0:
                    continue

                # append text, label, and spans:
                if add_spans: self.spans.append(s)
                self.X.append(text[start:stop])
                self.y.append(label)
                self.data_size += 1

        self.tokenizer = tokenizer
        self.add_texts = add_texts

    def __getitem__(self, index:int) -> tuple:
        # create a list of tokens:
        result = self.tokenizer(self.X[index], return_offsets_mapping=True, truncation=True, padding='max_length')

        # add labels and texts:
        if self.add_texts: result['texts'] = self.X[index]
        result['labels'] = self.y[index]

        # create map of important spans:
        offset_mapping = result.pop('offset_mapping')
        if self.spans is not None:
            result['spans'] = np.zeros(len(result['input_ids']), dtype=int)

            for span in self.spans[index]:
                for i, token in enumerate(offset_mapping):
                    for j in token:
                        if span.start <= j and span.stop > j:
                            result['spans'][i] = True

        for key in result:
            try: result[key] = torch.tensor(result[key])
            except TypeError: pass

        return result

    def __len__(self) -> int:
        return self.data_size

####################################################################################################
# Data I/O:                                                                                        #
####################################################################################################

def load_data(
        dir:str,
        text_column:str,
        label_column:str,
        split:int,
        tokenizer:PreTrainedTokenizerBase,
        label2int:Optional[Callable[[str],int]]=None,
        add_spans:bool=True,
        add_texts:bool=False,
        **kwargs
    ):
    # load data:
    with open(dir + f"split_{label_column.split('-')[0]}_{split:d}.pickle", "rb") as f:
        data = pickle.load(f)

    # convert to dataset class:
    data_train = ClassificationDataset(
        data['train'], text_column, label_column, tokenizer, label2int=label2int, **kwargs)
    data_valid = ClassificationDataset(
        data['valid'], text_column, label_column, tokenizer, label2int=label2int, add_spans=add_spans, add_texts=add_texts, **kwargs)
    data_test = ClassificationDataset(
        data['test'], text_column, label_column, tokenizer, label2int=label2int, add_spans=add_spans, add_texts=add_texts, **kwargs)

    # extract labels:
    keys_train = tuple([key for key in data_train[0]])
    keys_valid = tuple([key for key in data_valid[0]])

    # extract labels and texts:
    return (
        (data_train, keys_train),
        (data_valid, keys_valid),
        (data_test, keys_valid)
    )

def save_data(dir:str, split:int, task:str, data_train:pd.DataFrame, data_valid:pd.DataFrame, data_test:pd.DataFrame):
    # save data:
    with open(dir + f"split_{task}_{split:d}.pickle", "wb") as f:
        pickle.dump({'train':data_train,
                     'valid':data_valid,
                     'test':data_test}, f)

def load_mappings(dir:str, label:str):
    # load mappings:
    with open(dir + "mappings.pickle", "rb") as f:
        mappings = pickle.load(f)

    # extract mappings for label:
    return mappings[label]

def save_mappings(dir:str, products:Iterable[str], hazards:Iterable[str], productCategories:Iterable[str], hazardCategories:Iterable[str]):
    # save mappings:
    with open(dir + "mappings.pickle", "wb") as f:
        pickle.dump({'product':products,
                     'hazard':hazards,
                     'product-category':productCategories,
                     'hazard-category':hazardCategories}, f)