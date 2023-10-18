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

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA

####################################################################################################
# Type hints:                                                                                      #
####################################################################################################

import numpy.typing as npt
from typing import Tuple, Iterable, Union, Generator, Dict, Any
from sklearn.base import ClassifierMixin

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
        'n_neighbors':[2,4,6],
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

def _calculate_tf(i, t, input_ids):
    # count words:
    row = np.empty(len(input_ids), dtype=float)
    for j, doc in enumerate(input_ids):
        row[j] = (doc == t).sum()
    return i, row

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

class EvaluatorTfIdf(Evaluator):
    def __init__(self, normalize_fcn:T_norm=None, num_workers:int=10, _num_labels:int=0) -> None:
        super().__init__(normalize_fcn=normalize_fcn, num_labels=_num_labels)
        self._num_workers   = num_workers
        self._terms         = None
        self._tfidf         = None

    def _enumerate_data(self, data:T_data, filter_by_spans:bool=False) -> Generator[Tuple[npt.NDArray, npt.NDArray, float], None, None]:
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
            
            yield input_ids, label, weight

    def _encode_data(self, data:T_data, filter_by_spans:bool=False) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        n = len(data[0] if isinstance(data, tuple) else data)
        
        x = np.zeros((n, self._tfidf.shape[1]), dtype=float)
        y = np.empty((n, self._num_labels), dtype=int)
        w = np.empty(n, dtype=float)

        for i, (input_ids, label, weight) in enumerate(self._enumerate_data(data, filter_by_spans=filter_by_spans)):
            # create tf-idf-vector:
            for t in input_ids:
                if t in self._terms:
                    x[i] += self._tfidf[self._terms[t]]

            # normalize tf-idf-vector:
            x[i] = np.nan_to_num(x[i] / np.linalg.norm(x[i]))

            y[i] = label
            w[i] = weight

        return x, y, w

    @staticmethod
    def load(dir:str, normalize_fcn:T_norm=None, num_workers:int=10, **kwargs) -> Tuple['EvaluatorTfIdf', WordTokenizer]:
        '''Loads a model from disk.

            dir:           Path to the directory that contains the model (e.g.: .\models)

            normalize_fcn: Normalization function used on the predictions [`"max"` | `"sum"` | `"min-max"` | `"softmax"` | `None`]

            num_workers:   Number of threads for parallel processsing [`int`, default:10]

            
            returns:       Tuple (`EvaluatorTfIdf`, `WordTokenizer`) of the model and the corresponding tokenizer
        '''
        # create model instance:
        tf_idf = EvaluatorTfIdf(normalize_fcn=normalize_fcn, num_workers=num_workers)

        # load data from file:
        with open(dir + '/tf-idf.pickle', 'rb') as file:
            data = pickle.load(file)

        # update model parameters:
        tf_idf._num_labels = data['num_labels']
        tf_idf._terms      = data['token2index']
        tf_idf._tfidf      = data['matrix']
        tf_idf._model      = data['models']

        return tf_idf, WordTokenizer.load(dir, **kwargs)

    def save(self, dir:str, tokenizer:WordTokenizer) -> None:
        '''Saves a model to disk.

            dir:           Path to the directory that contains the model (e.g.: .\models)

            tokenizer:     The tokenizer object used on the training data
        '''
        # create directories:
        os.makedirs(dir, exist_ok=True)

        # save model:
        with open(dir + '/tf-idf.pickle', 'wb') as file:
            pickle.dump({
                'num_labels':  self._num_labels,
                'token2index': self._terms,
                'matrix':      self._tfidf,
                'models':      self._model
            }, file)

        # save tokenizer:
        tokenizer.save(dir)

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

    def cosine_similarity(self, a:str, b:str, tokenizer:WordTokenizer) -> float:
        v = np.zeros((2, self._tfidf.shape[1]), dtype=float)

        for i, text in enumerate([a,b]):
            # create tf-idf-vector:
            for t in tokenizer(text, return_offsets_mapping=False)['input_ids']:
                if t in self._terms:
                    v[i] += self._tfidf[self._terms[t]]

            # normalize tf-idf-vector:
            v[i] = np.nan_to_num(v[i] / np.linalg.norm(v[i]))

        # return dot product:
        return v.prod(axis=0).sum()

####################################################################################################
# Trainer Class:                                                                                   #
####################################################################################################

class TrainerTfIdf(EvaluatorTfIdf, Trainer):
    def __init__(self,
            num_labels:int,
            data_train:T_data,
            normalize_fcn:T_norm=None,
            num_workers:int=10,
            reduce_dim:bool=False,
            filter_by_spans:bool=False
        ) -> None:
        super().__init__(normalize_fcn=normalize_fcn, num_workers=num_workers, _num_labels=num_labels)

        # unpack data:
        x = [input_ids for input_ids, _, _ in self._enumerate_data(data_train, filter_by_spans=filter_by_spans)]

        # get unique tokens:
        self._terms = np.unique([token for entry in x for token in entry])
        self._terms.sort()
        self._terms = {key:i for i, key in enumerate(self._terms)}
        n_terms = len(self._terms)
        n_docs  = len(x)

        # calculate term frequency per window:
        tf = np.empty([n_terms, n_docs], dtype=int)
        queue = ArgumentQueue(_calculate_tf, list(enumerate(self._terms)), desc='Fitting tf-idf')
        for i, row in queue(n_workers=self._num_workers, input_ids=x): tf[i] = row
        #print(tf.shape)

        # calculate inverse document frequency per window:
        idf = np.nan_to_num(-np.log((tf > 0).mean(axis=-1, dtype=float)))
        #print(idf.shape)

        # calculate tf-idf:
        self._tfidf = tf*idf.reshape((-1,1))
        #print(self._tfidf.shape)

        if reduce_dim:
            oldshape = self._tfidf.shape
            print('\nReducing dimensionality...')
            pca = PCA(n_components='mle')
            self._tfidf = pca.fit_transform(self._tfidf)
            print(f'Done. Reduced dimensionality from {oldshape} to {self._tfidf.shape}.')

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
        model_name:        Name of the model to be trained (e.g.: "svm")

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
        model_path = f"models/tf-idf-{model_name}/tf-idf-{model_name}-{label_column}-{i:d}"
        result_path = f"results/tf-idf-{model_name}/tf-idf-{model_name}-{label_column}-{i:d}"

        # load data:
        data_train, data_valid, data_test = load_data(
            f"data/{dataset_name}/splits/",
            text_column,
            label_column,
            i,
            tokenizer
        )

        if not (predict or eval_explanations):
            # create trainer:
            trainer = TrainerTfIdf(
                len(label_map),
                data_train,
                normalize_fcn=normalize_fcn,
                num_workers=threads
            )

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
            trainer, tokenizer = EvaluatorTfIdf.load(
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