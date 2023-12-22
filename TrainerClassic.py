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
from resources.embedding import Embedding, EmbeddingBOW, EmbeddingTfIdf

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

class EvaluatorClassic(Evaluator):
    def __init__(self, embedding:Embedding=None, model:ClassifierMixin=None, num_labels:int=0, normalize_fcn:T_norm=None, num_workers:int=10) -> None:
        super().__init__(model=model, num_labels=num_labels, normalize_fcn=normalize_fcn)
        self._embedding     = embedding
        self._num_workers   = num_workers

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
    def embedding(self) -> Embedding:
        return self._embedding

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
    def load(dir:str, normalize_fcn:T_norm=None, num_workers:int=10, **kwargs) -> 'EvaluatorClassic':
        '''Loads a model from disk.

            dir:           Path to the directory that contains the model (e.g.: .\models)

            normalize_fcn: Normalization function used on the predictions [`"max"` | `"sum"` | `"min-max"` | `"softmax"` | `None`]

            num_workers:   Number of threads for parallel processsing [`int`, default:10]

            
            returns:       `EvaluatorClassic` object
        '''
        # load data from file:
        with open(dir + '/model.pickle', 'rb') as file:
            data = pickle.load(file)

        # create evaluator instance:
        return EvaluatorClassic(
            model         = data['model'],
            embedding     = data['embedding_type'].load(dir, **kwargs),
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
                'model':         self._model,
                'num_labels':     self._num_labels,
                'embedding_type': type(self._embedding)
            }, file)

        # save embedding:
        self._embedding.save(dir, **kwargs)

####################################################################################################
# Trainer Class:                                                                                   #
####################################################################################################

class TrainerClassic(EvaluatorClassic, Trainer):
    def __init__(self,
            embedding:Embedding,
            num_labels:int,
            normalize_fcn:T_norm=None,
            num_workers:int=10
        ) -> None:
        super().__init__(embedding=embedding, num_labels=num_labels, normalize_fcn=normalize_fcn, num_workers=num_workers)

    def __call__(self,
            model:str,
            data_train:T_data,
            data_valid:T_data,
            bias:float=0.5,
            modelargs:Dict[str, Iterable[Any]]={}
        ) -> None:
        best_score = -1.
        best_model = None

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
                best_model = self._model

        # switch to model with best parameters:
        self._model = best_model

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
        model_dir = f"models/{embedding_name}-{model_name}/{embedding_name}-{model_name}-{label_column}-{i:d}"
        result_dir = f"results/{embedding_name}-{model_name}"

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

            # create trainer:
            trainer = TrainerClassic(
                embedding,
                len(label_map),
                normalize_fcn=normalize_fcn,
                num_workers=threads
            )

            # compress encoding:
            if pca: trainer.embedding.fit_pca(data_train, 0.5)

            # train model:
            trainer(
                model[0],
                data_train,
                data_valid,
                modelargs=model[1]
            )

            # save trained model:
            trainer.save(dir=model_dir, tokenizer=tokenizer)

            # save history:
            with open(model_dir + '/hist.json', 'w') as file:
                file.write(str(trainer.train_history))

        else:
            # load model:
            trainer = EvaluatorClassic.load(
                dir=model_dir,
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
        help='Name of the model to be trained (e.g.: "bow-svm")'
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