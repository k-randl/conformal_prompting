# imports:
import os
import sys
import json
import pickle

import numpy as np

from resources.evaluator import Trainer, T_norm
from resources.data_io import load_data, load_mappings, T_data
from resources.tokenization import WordTokenizer
from resources.embedding import Embedding, EmbeddingBOW, EmbeddingTfIdf

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score
from sklearn.base import ClassifierMixin

####################################################################################################
# Type hints:                                                                                      #
####################################################################################################

import numpy.typing as npt
from typing import Iterable, Tuple, Type, Dict, Any

####################################################################################################
# Models:                                                                                          #
####################################################################################################

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from resources.models import SupportModel, RandomModel

MODELS = { 
    "rnd": (RandomModel, {}),
    "sup": (SupportModel, {}),
    "svm": (SVC, {
        'C':[0.5,1.0,2.0],
        'kernel':['linear'],
        'probability':[True],
        'verbose':[True]
    }),
    "knn": (KNeighborsClassifier, {
        'n_neighbors':[2,4,8],
        'metric':['l1', 'l2'],
        'n_jobs':[-1]
    }),
    "lr": (LogisticRegression, {
        'C':[0.5,1.0,2.0],
        'penalty':['l1', 'l2'],
        'solver':['liblinear'],
        'n_jobs':[-1],
        'verbose':[True]
    })
}

####################################################################################################
# Trainer Class:                                                                                   #
####################################################################################################

class TrainerClassic(Trainer):
    def __init__(self, num_labels:int=0, normalize_fcn:T_norm=None) -> None:
        super().__init__(num_labels=num_labels, normalize_fcn=normalize_fcn)
        self._embedding = None

    def _encode_data(self, data:T_data, filter_by_spans:bool=False) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        x, y, w = [], [], []

        # unpack keys:
        keys = ('input_ids', 'labels')
        if isinstance(data, tuple):
            data, keys = data

        # iterate through batches:
        self._last_spans = []
        self._last_texts = []
        for entry in data:
            # get new batch to gpu:
            label, input_ids, weight, spans, text = None, None, 1., None, None
            for key in keys:
                if   key == 'labels':    label     = entry[key].detach().numpy()
                elif key == 'input_ids': input_ids = entry[key].detach().numpy()
                elif key == 'weights':   weight    = entry[key].detach().numpy()
                elif key == 'spans':     spans     = entry[key].detach().numpy()
                elif key == 'texts':     text      = entry[key]

            # deal with texts:
            if text is not None:
                self._last_texts.append(text)

            # deal with spans:
            if spans is not None:
                self._last_spans.append(spans)
                    
                # filter by spans:
                if filter_by_spans:
                    input_ids = input_ids[spans]
            
            x.append(input_ids)
            y.append(label)
            w.append(weight)

        # encode and return data:
        return self._embedding(x), np.array(y, dtype=int), np.array(w, dtype=float)

    @property
    def tokenizer(self) -> WordTokenizer:
        if self._embedding is None: return None
        else: return self._embedding.tokenizer

    @property
    def embedding(self) -> Embedding:
        return self._embedding

    def predict(self, data:T_data, output_probabilities:bool=True, output_spans:bool=False) -> Dict[str, Any]:
        result = {}

        # get data:
        x, result['labels'], _ = self._encode_data(data)

        if output_probabilities:
            # create probability array:
            p = np.zeros((len(x), self._num_labels), dtype=float)

            # predict probabilities:
            p[:,self._model.classes_] = self._model.predict_proba(x)

            # save most probable class as prediction:
            result['predictions']   = np.argmax(p, axis=-1)
            result['probabilities'] = np.apply_along_axis(self._normalize_fcn, 1, p)

        # otherwise directly predict labels:
        else: result['predictions'] = self._model.predict(x)

        # add spans to list:
        if output_spans: result['spans'] = self._last_spans

        return result

    @staticmethod
    def load(dir:str, normalize_fcn:T_norm=None, num_workers:int=10, **kwargs) -> 'TrainerClassic':
        '''Loads a model from disk.

            dir:           Path to the directory that contains the model (e.g.: .\models)

            normalize_fcn: Normalization function used on the predictions [`"max"` | `"sum"` | `"min-max"` | `"softmax"` | `None`]

            num_workers:   Number of threads for parallel processsing [`int`, default:10]

            
            returns:       `TrainerClassic` object
        '''
        # load data from file:
        with open(dir + '/model.pickle', 'rb') as file:
            data = pickle.load(file)

        # create evaluator instance:
        trainer =  TrainerClassic(
            num_labels    = data['num_labels'],
            normalize_fcn = normalize_fcn
        )
        trainer._model     = data['model']
        trainer._embedding = data['embedding_type'].load(dir, **kwargs)

        return trainer

    def save(self, dir:str, **kwargs) -> None:
        '''Saves a model to disk.

            dir:           Path to the directory that contains the model (e.g.: .\models)
        '''
        # create directories:
        os.makedirs(dir, exist_ok=True)

        # save model:
        with open(dir + '/model.pickle', 'wb') as file:
            pickle.dump({
                'model':          self._model,
                'num_labels':     self._num_labels,
                'embedding_type': type(self._embedding)
            }, file)

        # save embedding:
        self._embedding.save(dir, **kwargs)

    def fit(self, embedding:Embedding, model:Type[ClassifierMixin], data_train:T_data, data_valid:T_data, modelargs:Dict[str, Iterable[Any]]={}) -> None:
        best_score = -1.
        best_model = None

        # create embedding:
        self._embedding = embedding

        # unpack data:
        x, y, w = self._encode_data(data_train)

        self._train_history = []
        for kwargs in ParameterGrid(modelargs):
            # fit model with new parameters:
            self._model = model(**kwargs)
            try: self._model.fit(x, y, w)
            except: self._model.fit(x, y)

            # predict model:
            valid_out = self.predict(data_valid, output_probabilities=False)

            # calculate score:
            y_true = valid_out['labels'].astype(int)
            y_pred = valid_out['predictions'].astype(int)
            score  = f1_score(y_true, y_pred, average='macro')

            #save score to history:
            hist_item = kwargs.copy()
            hist_item['score'] = score
            self._train_history.append(hist_item)

            if score > best_score:
                print(f'New best score: {score:.2f} (args={str(kwargs)})')
                best_score = score
                best_model = self._model

        # switch to model with best parameters:
        self._model = best_model

####################################################################################################
# Main-Function:                                                                                   #
####################################################################################################

def run(model_name:str, text_column:str, label_column:str, dataset_name:str,
        iterations:        Iterable[int]   = range(5),
        normalize_fcn:     T_norm          = None,
        pca:               bool            = False,
        train:             bool            = False,
        predict:           bool            = False,
        eval_explanations: bool            = False
    ) -> None:
    '''Train and/or Evaluate a baseline model.

    arguments:
        model_name:        Name of the model to be trained (e.g.: "bow-svm")

        text_column:       Name of the column to be used as the model's input

        label_column:      Name of the column to be used as the label

        dataset_name:      Name of the dataset (e.g.: "incidents")

        iterations:        K-Fold iterations (default:[0,1,2,3,4,5])

        normalize_fcn:     Normalization applied on the results after prediction (default:None)

        pca:               Use Principal Component Analysis for reducing the embeding dimensions (default:False)

        train:             Only train model (default:False)

        predict:           Only predict test set (default:False)

        eval_explanations: Evaluate attentions against spans (default:False)
    '''
    embedding_name, model_name = tuple(model_name.split('-')) 

    try: model = MODELS[model_name.lower()]
    except KeyError: raise ValueError(f'Unknown model "{model_name}"!')

    # load mappings:
    label_map = load_mappings(f"data/{dataset_name}/splits/", label_column)

    for i in iterations:
        print("\n----------------------------------------------------------------------------------------------------")
        print(f" {model_name}: iteration {i:d}")
        print(f" (label = '{label_column}')")
        print("----------------------------------------------------------------------------------------------------\n")

        # define paths:
        model_dir = f"models/{embedding_name}-{model_name}/{embedding_name}-{model_name}-{label_column}-{i:d}"
        result_dir = f"results/{embedding_name}-{model_name}"

        if not (predict or eval_explanations):
            # create tokenizer:
            tokenizer = WordTokenizer()

            # load data:
            data_train, data_valid, data_test = load_data(
                f"data/{dataset_name}/splits/",
                text_column,
                label_column,
                i,
                tokenizer
            )

            # create embedding:
            if embedding_name == "bow":
                embedding = EmbeddingBOW(tokenizer)
                for _ in data_train[0]: pass
                tokenizer.train = False

            elif embedding_name == "tfidf":
                embedding = EmbeddingTfIdf(
                    [entry['input_ids'].detach().numpy() for entry in data_train[0]],
                    tokenizer
                )
                tokenizer.train = False

            else: raise ValueError(f'Unknown embedding "{embedding_name}"!')

            # compress encoding:
            if pca: embedding.fit_pca(data_train, 0.5)

            # create trainer:
            trainer = TrainerClassic(
                len(label_map),
                normalize_fcn=normalize_fcn
            )

            # train model:
            trainer.fit(
                embedding,
                model[0],
                data_train,
                data_valid,
                modelargs=model[1]
            )

            # save trained model:
            trainer.save(dir=model_dir, tokenizer=tokenizer)

            # save history:
            with open(model_dir + '/hist.json', 'w') as f:
                json.dump(trainer.train_history, f)

        else:
            # load model:
            trainer = TrainerClassic.load(
                dir=model_dir,
                normalize_fcn=normalize_fcn
            )

            # load data:
            _, _, data_test = load_data(
                f"data/{dataset_name}/splits/",
                text_column,
                label_column,
                i,
                trainer.tokenizer
            )

        if not (train or eval_explanations):
            # evaluate model:
            results = trainer.predict(
                data_test,
                output_spans=False
            )

            # save predictions:
            os.makedirs(result_dir, exist_ok=True)
            with open(f'{result_dir}/{embedding_name}-{model_name}-{label_column}-{i:d}.pickle', 'wb') as f:
                pickle.dump(results, f)

        if eval_explanations:
            raise NotImplementedError()

####################################################################################################
# Main Function:                                                                                   #
####################################################################################################

if __name__ == '__main__':
    from argparse import ArgumentParser

    # create parser:
    parser = ArgumentParser(description='Train and/or Evaluate a baseline model.')
    parser.add_argument('model_name',
        type=str,
        help='name of the model to be trained (e.g.: "bow-svm")'
    )
    parser.add_argument('text_column',
        type=str,
        help='name of the column to be used as the model\'s input',
    )
    parser.add_argument('label_column',
        type=str,
        help='name of the column to be used as the label',
    )
    parser.add_argument('dataset_name',
        type=str,
        help='name of the dataset (e.g.: "incidents")'
    )
    parser.add_argument('-i', '--iterations',
        metavar='I',
        nargs='+',
        type=int,
        default=list(range(5)),
        help='k-fold iterations'
    )
    parser.add_argument('-nf', '--normalize_fcn',
        metavar='NF',
        type=str,
        default=None,
        help='normalization applied on the results after prediction (default: no normalization)'
    )
    parser.add_argument('--pca',
        action='store_true',
        help = 'use principal component analysis for reducing the embeding dimensions'
    )
    parser.add_argument('--train',
        action='store_true',
        help='only train model'
    )
    parser.add_argument('--predict',
        action='store_true',
        help='only predict test set'
    )
    parser.add_argument('--eval_explanations',
        action='store_true',
        help='evaluate attentions against spans'
    )

    # parse arguments:
    args = parser.parse_args()

    # run main function:
    sys.exit(run(**dict(args._get_kwargs())))