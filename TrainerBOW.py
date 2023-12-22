# imports:
import os
import sys
import pickle

import numpy as np

from tqdm.autonotebook import tqdm

from resources.evaluator import Evaluator, Trainer, T_norm
from resources.data_io import load_data, load_mappings, T_data
from resources.multiprocessing import ArgumentQueue
from resources.tokenization import WordTokenizer
from resources.embedding import Emdedding, EmdeddingBOW, EmdeddingTfIdf

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score
from sklearn.base import ClassifierMixin

####################################################################################################
# Type hints:                                                                                      #
####################################################################################################

import numpy.typing as npt
from typing import Iterable, Tuple, Dict, Any

####################################################################################################
# Models:                                                                                          #
####################################################################################################

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from resources.models import SupportModel, RandomModel, DummyModel

MODELS = { 
    "rnd": (RandomModel, {}),
    "sup": (SupportModel, {}),
    "svm": (SVC, {
        'C':[0.5,1.0,2.0],
        'kernel':['linear'],
        'probability':[True]
    }),
    "knn": (KNeighborsClassifier, {
        'n_neighbors':[2,4,8],
        'metric':['l1', 'l2'],
#        'n_jobs':[-1]
    }),
    "lr": (LogisticRegression, {
        'C':[0.5,1.0,2.0],
        'penalty':['l1', 'l2'],
        'solver':['liblinear'],
#        'verbose':[1],
#        'n_jobs':[-1]
    })
}

####################################################################################################
# Thread Functions:                                                                                #
####################################################################################################

def _train_model(i, data, model, modelargs):
    x, y, w = data

    labels = np.unique(y[:,i])
                
    if len(labels) > 1:
        m = model(**modelargs)
        try: m.fit(x, y[:,i], w)
        except: m.fit(x, y[:,i])

        return i, m

    else: return i, DummyModel(labels[0])

def _predict_class(i, m, x):
    return i, m.predict_proba(x)[:,m.classes_[1]]

####################################################################################################
# Evaluator Class:                                                                                 #
####################################################################################################

class EvaluatorEmdedding(Evaluator):
    def __init__(self, emdedding:Emdedding=None, model:ClassifierMixin=None, num_labels:int=0, normalize_fcn:T_norm=None, num_workers:int=10) -> None:
        super().__init__(model=model, num_labels=num_labels, normalize_fcn=normalize_fcn)
        self._emdedding     = emdedding
        self._num_workers   = num_workers

    def _encode_data(self, data:T_data, filter_by_spans:bool=False) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        # enumerate data:
        x, y, w = [], [], []
        for input_ids, label, weight in enumerate(self._enumerate_data(data, filter_by_spans=filter_by_spans)):
            x.append(input_ids)
            y.append(label)
            w.append(weight)

        # encode and return data:
        return self._emdedding(x), np.array(y, dtype=int), np.array(w, dtype=float)

    @property
    def emdedding(self) -> Emdedding:
        return self._emdedding

    def predict(self, data:T_data, output_spans:bool=False) -> Dict[str, Any]:
        x, y_true, _ = self._encode_data(data)
        
        y_pred = np.zeros((len(x), self._num_labels), dtype=float)
#        queue = ArgumentQueue(
#            _predict_class,
#            list(enumerate(self._model)),
#            desc="Evaluation"
#        )
#        for i, y in queue(n_workers=self._num_workers, x=x): y_pred[:,i] = y
        for i, m in tqdm(enumerate(self._model), total=self._num_labels, desc="Evaluation"):
            y_pred[:,i] = m.predict_proba(x)[:,m.classes_[1]]

        # create and return output dictionary:
        result = {
            'labels':np.array(y_true),
            'predictions':np.apply_along_axis(self._normalize_fcn, 1, y_pred)
        }
        # add spans to list:
        if output_spans: result['spans'] = self._last_spans
        return result

    @staticmethod
    def load(dir:str, normalize_fcn:T_norm=None, num_workers:int=10, **kwargs) -> 'EvaluatorEmdedding':
        '''Loads a model from disk.

            dir:           Path to the directory that contains the model (e.g.: .\models)

            normalize_fcn: Normalization function used on the predictions [`"max"` | `"sum"` | `"min-max"` | `"softmax"` | `None`]

            num_workers:   Number of threads for parallel processsing [`int`, default:10]

            
            returns:       `EvaluatorEmdedding` object
        '''
        # load data from file:
        with open(dir + '/model.pickle', 'rb') as file:
            data = pickle.load(file)

        # create evaluator instance:
        return  EvaluatorEmdedding(
            model         = data['models'],
            emdedding       = data['emdedding_type'].load(dir, **kwargs),
            num_labels    = data['num_labels'],
            normalize_fcn = normalize_fcn,
            num_workers   = num_workers
        )

    def save(self, dir:str, **kwargs) -> None:
        '''Saves a model to disk.

            dir:           Path to the directory that contains the model (e.g.: .\models)
        '''
        # create directories:
        os.makedirs(dir, exist_ok=True)

        # save model:
        with open(dir + '/model.pickle', 'wb') as file:
            pickle.dump({
                'models':       self._model,
                'num_labels':   self._num_labels,
                'emdedding_type': type(self._emdedding)
            }, file)

        # save emdedding:
        self._emdedding.save(dir, **kwargs)

####################################################################################################
# Trainer Class:                                                                                   #
####################################################################################################

class TrainerEmdedding(EvaluatorEmdedding, Trainer):
    def __init__(self,
            embedding:Emdedding,
            num_labels:int,
            normalize_fcn:T_norm=None,
            num_workers:int=10
        ) -> None:
        super().__init__(emdedding=embedding, num_labels=num_labels, normalize_fcn=normalize_fcn, num_workers=num_workers)

    def __call__(self,
            model:str,
            data_train:T_data,
            data_valid:T_data,
            bias:float=0.5,
            modelargs:Dict[str, Iterable[Any]]={}
        ) -> None:
        best_score = -1.
        best_models = None

        # unpack data:
        x, y, w = self._encode_data(data_train)

        self._train_history = []
        for kwargs in ParameterGrid(modelargs):
            # fit model with new parameters:
            self._model = [None]*self._num_labels
            queue = ArgumentQueue(_train_model, list(zip(range(self._num_labels))), desc=f'Training classifier ({str(kwargs)})')
            for i, m in queue(n_workers=self._num_workers, data=(x, y, w), model=model, modelargs=kwargs): self._model[i] = m

            # predict model:
            valid_out = self.predict(data_valid)

            # calculate score:
            y_true = (valid_out['labels']).astype(int)
            y_pred = (valid_out['predictions'] > bias).astype(int)
            score  = f1_score(y_true, y_pred, average='macro')

            #save score to history:
            hist_item = kwargs.copy()
            hist_item['score'] = score
            self._train_history.append(hist_item)

            if score > best_score:
                print(f'New best score: {score:.2f} (args={str(kwargs)}')
                best_score  = score
                best_models = self._model

        # switch to model with best parameters:
        self._model = best_models

####################################################################################################
# Main-Function:                                                                                   #
####################################################################################################

def run(model_name:str, text_column:str, label_column:str, dataset_name:str,
        iterations:        Iterable[int]   = range(5),
        normalize_fcn:     T_norm          = 'min-max',
        threads:           int             = 10,
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

        normalize_fcn:     Normalization applied on the results after prediction (default:'min-max')

        threads:           Number of threads for parallel processsing (default:10)

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

    # load tokenizer:
    tokenizer = WordTokenizer()

    for i in iterations:
        print("\n----------------------------------------------------------------------------------------------------")
        print(f" {model_name}: iteration {i:d}")
        print(f" (label = '{label_column}')")
        print("----------------------------------------------------------------------------------------------------\n")

        # define paths:
        model_path = f"models/{embedding_name}-{model_name}/{embedding_name}-{model_name}-{label_column}-{i:d}"
        result_path = f"results/{embedding_name}-{model_name}/{embedding_name}-{model_name}-{label_column}-{i:d}"

        # load data:
        data_train, data_valid, data_test = load_data(
            f"data/{dataset_name}/splits/",
            text_column,
            label_column,
            i,
            tokenizer
        )

        if not (predict or eval_explanations):
            # create embedding:
            if embedding_name == "bow":     embedding = EmdeddingBOW(tokenizer)
            elif embedding_name == "tfidf": embedding = EmdeddingTfIdf(data_train, tokenizer)

            else: raise ValueError(f'Unknown embedding "{embedding_name}"!')

            # create trainer:
            trainer = TrainerEmdedding(
                embedding,
                len(label_map),
                normalize_fcn=normalize_fcn,
                num_workers=threads
            )

            # compress encoding:
            if pca: trainer.emdedding.fit_pca(data_train, 0.5)

            # train model:
            trainer(
                model[0],
                data_train,
                data_valid,
                modelargs=model[1]
            )

            # save trained model:
            trainer.save(dir=model_path, tokenizer=tokenizer)

            # save history:
            with open(model_path + '/hist.json', 'w') as file:
                file.write(str(trainer.train_history))

        else:
            # load model:
            trainer, tokenizer = EvaluatorEmdedding.load(
                dir=model_path,
                normalize_fcn=normalize_fcn,
                num_workers=threads
            )

        if not (train or eval_explanations):
            # evaluate model:
            results = trainer.predict(
                data_test,
                output_spans=False
            )

            # save predictions:
            with open(f"{result_path}.pickle", "wb") as f:
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
        help='Name of the model to be trained (e.g.: "svm")'
    )
    parser.add_argument('text_column',
        type=str,
        help='Name of the column to be used as the model\'s input',
    )
    parser.add_argument('label_column',
        type=str,
        help='Name of the column to be used as the label',
    )
    parser.add_argument('dataset_name',
        type=str,
        help='Name of the dataset (e.g.: "incidents")'
    )
    parser.add_argument('--iterations',
        metavar='-i',
        nargs='+',
        type=int,
        default=list(range(5)),
        help='K-Fold iterations'
    )
    parser.add_argument('--normalize_fcn',
        metavar='-nf',
        type=str,
        default='min-max',
        help='Normalization applied on the results after prediction (default:\'min-max\')'
    )
    parser.add_argument('--threads',
        metavar='-t',
        type=int,
        default=10,
        help='Number of threads for parallel processsing'
    )
    parser.add_argument('--pca',
        action='store_true',
        help = 'Use Principal Component Analysis for reducing the embeding dimensions'
    )
    parser.add_argument('--train',
        action='store_true',
        help='Only train model'
    )
    parser.add_argument('--predict',
        action='store_true',
        help='Only predict test set'
    )
    parser.add_argument('--eval_explanations',
        action='store_true',
        help='Evaluate attentions against spans'
    )

    # parse arguments:
    args = parser.parse_args()

    # run main function:
    sys.exit(run(**dict(args._get_kwargs())))