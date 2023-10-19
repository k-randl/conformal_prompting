# imports:
import os
import sys
import pickle
import shutil

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, ConstantLR, LinearLR
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedModel

import numpy as np

from tqdm import trange
from tqdm.autonotebook import tqdm

from resources.evaluator import Evaluator, Trainer, T_norm
from resources.data_io import load_data, load_mappings, T_data
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score


####################################################################################################
# Train on gpu if possible:                                                                        #
####################################################################################################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####################################################################################################
# Type hints:                                                                                      #
####################################################################################################

import numpy.typing as npt
from typing import Optional, Callable, Tuple, Iterable, Union, Generator, Dict, Any

####################################################################################################
# Evaluator Class:                                                                                 #
####################################################################################################

class EvaluatorTransformer(Evaluator):
    def __init__(self, batch_size:int=None, normalize_fcn:T_norm=None, loss_fcn:Optional[nn.Module]=None, _num_labels=0) -> None:
        super().__init__(normalize_fcn=normalize_fcn, num_labels=_num_labels)
        self._batch_size = batch_size
        self._loss_fcn   = loss_fcn

    def _init_model(self, model:str, **kwargs):
        model = AutoModelForSequenceClassification.from_pretrained(
            model,
            num_labels=self._num_labels,
            **kwargs
        )

        self._model = nn.DataParallel(model)
        self._model.to(DEVICE)
        print('Running on', DEVICE)

    def _enumerate_data(self, data:T_data, desc:str='', shuffle:bool=False) -> Generator[Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None], Dict[str, torch.Tensor]], None, None]:
        # unpack keys:
        keys = ('input_ids', 'labels')
        if isinstance(data, tuple):
            data, keys = data

        # create DataLoader
        if isinstance(data, Dataset):
            data = DataLoader(data,
                shuffle=shuffle,
                batch_size=self._batch_size
            )

        # iterate through batches:
        self._last_texts = []
        for batch in tqdm(data, desc=desc):
            # get new batch to gpu:
            labels, weights, spans, texts, kwargs = None, None, None, None, {}
            for key in keys:
                if   key == 'labels':  labels      = batch[key].to(DEVICE)
                elif key == 'weights': weights     = batch[key].to(DEVICE)
                elif key == 'spans':   spans       = batch[key].to(DEVICE)
                elif key == 'texts':   texts       = batch[key]
                else:                  kwargs[key] = batch[key].to(DEVICE)

            # deal with texts:
            if texts is not None:
                self._last_texts.extend(texts)

            yield labels, weights, spans, kwargs

    @property
    def device(self) -> torch.device:
        return DEVICE

    @property
    def last_attentions(self) -> Union[Iterable[npt.NDArray], npt.NDArray, None]:
        if hasattr('_last_attentions', self):
            return self._last_attentions
        else: return None

    @property
    def last_hidden_states(self) -> Union[Iterable[torch.Tensor], torch.Tensor, None]:
        if hasattr('_last_hidden_states', self):
            return self._last_hidden_states
        else: return None

    @staticmethod
    def load(dir:str, num_labels:int, batch_size:int=None, normalize_fcn:T_norm=None, loss_fcn:Optional[nn.Module]=None, **kwargs) -> 'EvaluatorTransformer':
        '''Loads a model from disk.

            dir:            Path to the directory that contains the model (e.g.: .\models)

            num_labels:     Number of unique labels in the data

            batch_size:     Batch size: 16, 32 [Devlin et al.]

            normalize_fcn:  Normalization function used on the predictions [`"max"` | `"sum"` | `"min-max"` | `"softmax"` | `None`]


            returns:        An instance of `EvaluatorTransformerTfIdf` with the model.
        '''
        evaluator = EvaluatorTransformer(
            batch_size=batch_size,
            normalize_fcn=normalize_fcn,
            loss_fcn=loss_fcn,
            _num_labels=num_labels
        )
        evaluator._init_model(dir, **kwargs)

        return evaluator

    def save(self, dir:str) -> None:
        '''Saves a model to disk.

            dir:           Path to the directory that contains the model (e.g.: .\models)
        '''
        # make sure that model is set:
        assert not (self._model is None)

        # create folder if necessary:
        os.makedirs(dir, exist_ok=True)

        # get model and state dict to be saved:
        model = self._model.module if hasattr(self._model, 'module') else self._model
        state_dict = model.state_dict()

        # if model is a PreTrainedModel-instance use transformers save format:
        if issubclass(type(model), PreTrainedModel):
            model.save_pretrained(dir, state_dict=state_dict)

        # otherwise save in torch style:
        else:
            torch.save(state_dict, f'{dir}/{type(model).__name__}.pt')

    def predict(self, data:T_data, output_spans:bool=False, output_attentions:bool=False, output_hidden_states:bool=False) -> Dict[str, any]:
        assert not (self._model is None)
        self._model.eval()

        eval_loss = 0
        n_steps = 0
        predictions, labels = [], []
        spans, attentions, hidden_states = [], [], []

        for labels_tensor, weights_tensor, spans_tensor, kwargs in self._enumerate_data(data, desc="Evaluation"):
            # if no loss fcn specified: use internal loss fcn of model
            if self._loss_fcn is None: kwargs['labels'] = labels_tensor

            # without gradient calculation:
            with torch.no_grad():
                # feed input through model:
                model_out = self._model(**kwargs)
                predictions_tensor = model_out['logits']

                if spans_tensor is not None:
                    self._last_spans = [t.detach().to('cpu').numpy() for t in spans_tensor]
                    if output_spans: spans.extend(self._last_spans)

                if hasattr(model_out, 'attentions'):
                    if model_out.attentions is not None:
                        self._last_attentions = [t.detach().to('cpu').numpy() for t in model_out.attentions]
                        if output_attentions: attentions.extend(self._last_attentions)

                if hasattr(model_out, 'hidden_states'):
                    if model_out.hidden_states is not None:
                        self._last_hidden_states = [t.detach().to('cpu').numpy() for t in model_out.hidden_states]
                        if output_hidden_states: hidden_states.extend(self._last_hidden_states)

                # add predictions and labels to lists:
                predictions.extend(list(predictions_tensor.to('cpu').numpy()))
                labels.extend(list(labels_tensor.to('cpu').numpy()))

                # update variables for mean loss calculation:
                if not (self._loss_fcn is None):
                    loss = self._loss_fcn(predictions_tensor.type(torch.float), labels_tensor.type(torch.float))
                    eval_loss += loss.item()
                    n_steps += 1

                elif hasattr(model_out, 'loss'):
                    loss = model_out.loss
                    eval_loss += loss.item()
                    n_steps += 1

        # create and return output dictionary:
        result = {
            'labels':np.array(labels),
            'predictions':np.apply_along_axis(self._normalize_fcn, 1, predictions)
        }
        if output_spans:            result['spans'] = spans
        if output_attentions:       result['attentions'] = attentions
        if output_hidden_states:    result['hidden_states'] = hidden_states
        if n_steps > 0:             result['loss'] = eval_loss/n_steps
        return result


####################################################################################################
# Trainer Class:                                                                                   #
####################################################################################################

class TrainerTransformer(EvaluatorTransformer, Trainer):
    def __init__(self,
            num_labels:int,
            batch_size:int,
            normalize_fcn:T_norm=None,
            model_dir:str='./model/',
            max_grad_norm:float=1.,
            loss_fcn:Optional[nn.Module]=None

        ) -> None:
        super().__init__(
            batch_size=batch_size,
            normalize_fcn=normalize_fcn,
            loss_fcn=loss_fcn,
            _num_labels=num_labels
        )
        self._model_dir = os.path.abspath(model_dir)
        self._tmp_dir=os.path.join(self._model_dir, '.tmp')
        self._max_grad_norm = max_grad_norm

    def __call__(self,
            model:str,
            data_train:T_data,
            data_valid:T_data,
            bias:float=0.5,
            modelargs: Dict[str, Iterable[Any]]={}
        ) -> None:
        best_f1   = float('-inf')
        best_loss = float('inf')

        self._train_history = []
        for kwargs in ParameterGrid(modelargs):
            # create new model: 
            self._init_model(model,
                output_attentions=False,
                output_hidden_states=False
            )

            # train model:
            f1, loss, loss_train, loss_valid = self.fit(
                data_train,
                data_valid,
                **kwargs
            ) 

            #save score to history:
            hist_item = kwargs.copy()
            hist_item['max_f1'] = f1
            hist_item['min_loss'] = loss
            hist_item['loss_train'] = loss_train
            hist_item['loss_valid'] = loss_valid
            self._train_history.append(hist_item)

            if f1 > best_f1:
                print(f'New best f1: {f1:.2f} (args={str(kwargs)}')
                best_f1 = f1
                shutil.copytree(
                    os.path.join(self._tmp_dir, 'f1'),
                    os.path.join(self._model_dir, 'f1'),
                    dirs_exist_ok=True
                )

            if loss < best_loss:
                print(f'New best loss: {loss:.2f} (args={str(kwargs)}')
                best_loss = loss
                shutil.copytree(
                    os.path.join(self._tmp_dir, 'loss'),
                    os.path.join(self._model_dir, 'loss'),
                    dirs_exist_ok=True
                )

        # clean up:
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def train(self, data:T_data, output_spans:bool=False,  output_attentions:bool=False, output_hidden_states:bool=False, shuffle:bool=False) -> Dict[str, any]:
        '''One epoch of training.'''
        self._model.train()

        train_loss = 0
        n_steps = 0

        spans, attentions, hidden_states = [], [], []

        for labels_tensor, weights_tensor, spans_tensor, kwargs in self._enumerate_data(data, desc="Training", shuffle=shuffle):
            # if no loss fcn specified: use internal loss fcn of model
            if self._loss_fcn is None: kwargs['labels'] = labels_tensor

            # feed input through model:
            model_out = self._model(**kwargs)

            if spans_tensor is not None:
                self._last_spans = [t.detach().to('cpu').numpy() for t in spans_tensor]
                if output_spans: spans.extend(self._last_spans)

            if hasattr(model_out, 'attentions'):
                if model_out.attentions is not None:
                    self._last_attentions = [t.detach().to('cpu').numpy() for t in model_out.attentions]
                    if output_attentions: attentions.extend(self._last_attentions)

            if hasattr(model_out, 'hidden_states'):
                if model_out.hidden_states is not None:
                    self._last_hidden_states = [t.detach().to('cpu').numpy() for t in model_out.hidden_states]
                    if output_hidden_states: hidden_states.extend(self._last_hidden_states)

            # calculate loss:
            loss = None
            if self._loss_fcn is not None:
                predictions_tensor = model_out['logits']
                loss = self._loss_fcn(predictions_tensor.type(torch.float), labels_tensor.type(torch.float))

            elif hasattr(model_out, 'loss'):
                loss = model_out.loss

            else: raise AttributeError("No loss function defined.")

            # update variables for mean loss calculation:
            train_loss += loss.item()
            n_steps += 1

            # backpropagate loss:
            loss.backward()

            # clip grads to avoid exploding gradients:
            nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)

            # step optimization:
            if self._optimizer != None:
                self._optimizer.step()
                self._optimizer.zero_grad()

        # step scheduler:
        if self._scheduler != None:
            self._scheduler.step()

        # create and return output dictionary:
        result = {}
        if output_spans:            result['spans'] = spans
        if output_attentions:       result['attentions'] = attentions
        if output_hidden_states:    result['hidden_states'] = hidden_states
        if n_steps > 0:             result['loss'] = train_loss/n_steps
        return result

    def fit(self, data_train:T_data, data_valid:T_data, lr:float=5e-5, epochs:int=1, patience:int=0, bias:float=.5, pretrain:bool=False, use_f1:bool=True, shuffle:bool=False):
        '''Model training for several epochs using early stopping.'''

        # update properties:
        self._optimizer = AdamW(self._model.parameters(), lr=lr)

        epochs_const = int(0.1 * epochs)
        s_const  = ConstantLR(self._optimizer, factor=1., total_iters=epochs_const)
        s_linear = LinearLR(self._optimizer, start_factor=1., end_factor=0.01, total_iters=epochs-epochs_const)
        self._scheduler = SequentialLR(self._optimizer,
            [s_const, s_linear],
            milestones=[epochs_const]
        )

        # lists for per epoch metrics:
        loss_train = np.array([np.NaN] * epochs, dtype=float)
        loss_valid = np.array([np.NaN] * epochs, dtype=float)
        f1_valid =   np.array([np.NaN] * epochs, dtype=float)

        # early stopping variables:
        best_loss_found = False
        best_f1_found = not use_f1

        # activate pretraining:
        if pretrain: self._model.set_pretrain()

        for i in trange(int(epochs), desc="Epoch"):
            # train one epoch using training data and predict on test data:
            train_out = self.train(data_train, shuffle=shuffle)
            valid_out = self.predict(data_valid)
            loss_train[i] = train_out['loss']
            loss_valid[i] = valid_out['loss']

            # print metrics:
            print(f"\nTraining:")
            print(f"  Loss:      {loss_train[i]:4.2f}")

            print(f"\nValidation:")
            print(f"  Loss:      {loss_valid[i]:4.2f}")

            if use_f1:
                y_true = np.array(valid_out['labels'], dtype=int)
                y_pred = np.array(valid_out['predictions'] > bias, dtype=int)
                f1_valid[i] = f1_score(y_true, y_pred, average='macro')
                print(f"  F1:        {f1_valid[i]:4.2f}")
                print(f"  Precision: {precision_score(y_true, y_pred, average='macro'):4.2f}")
                print(f"  Recall:    {recall_score(y_true, y_pred, average='macro'):4.2f}\n")

            # deactivate pretraining before saving:
            if pretrain: self._model.reset_pretrain()

            # save models in first round:
            if i == 0:
                if use_f1: self.save(os.path.join(self._tmp_dir, 'f1'))
                self.save(os.path.join(self._tmp_dir, 'loss'))

                # reactivate pretraining after saving:
                if pretrain: self._model.set_pretrain()
                continue

            # save models with best f1:
            if not best_f1_found:
                i_max_f1 = np.argmax(f1_valid[:i])
                if f1_valid[i] > f1_valid[i_max_f1]:
                    self.save(os.path.join(self._tmp_dir, 'f1'))
                elif i_max_f1 <= i - patience:
                    print(f"Early stopping for max. F1 initiated.\nBest epoch: {i_max_f1:d}\n")
                    best_f1_found = True

            # save models with best loss:
            if not best_loss_found:
                i_min_loss = np.argmin(loss_valid[:i])
                if loss_valid[i] < loss_valid[i_min_loss]:
                    self.save(os.path.join(self._tmp_dir, 'loss'))
                elif i_min_loss <= i - patience:
                    print(f"Early stopping for min. loss initiated.\nBest epoch: {i_min_loss:d}\n")
                    best_loss_found = True

            # stop training if done:
            if best_loss_found and best_f1_found: break

            # reactivate pretraining after saving:
            if pretrain: self._model.set_pretrain()

        if use_f1:  return np.nanmax(f1_valid), np.nanmin(loss_valid), loss_train, loss_valid
        else:       return float('-inf'), np.nanmin(loss_valid), loss_train, loss_valid

####################################################################################################
# Main-Function:                                                                                   #
####################################################################################################

def run(model_name:str, text_column:str, label_column:str, dataset_name:str,
        learning_rate:     float           = 5e-5,
        batch_size:        int             = 8,
        epochs:            int             = 4,
        patience:          int             = 1,
        iterations:        Iterable[int]   = range(5),
        normalize_fcn:     T_norm          = 'min-max',
        shuffle:           bool            = False,
        train:             bool            = False,
        predict:           bool            = False,
        eval_explanations: bool            = False
    ) -> None:
    '''Train and/or Evaluate a baseline model.

    arguments:
        model_name:        Name of the model to be trained (e.g.: "bert-base-uncased")

        text_column:       Name of the column to be used as the model's input

        label_column:      Name of the column to be used as the label

        dataset_name:      Name of the dataset (e.g.: "incidents")

        learning_rate:     Learning rate (Adam): 5e-5, 3e-5, 2e-5 [Devlin et al.]

        batch_size:        Batch size: 16, 32 [Devlin et al.]

        epochs:            Number of epochs: 2, 3, 4 [Devlin et al.]

        patience:          Patience for early stopping (0 means no early stopping)

        iterations:        K-Fold iterations (default:[0,1,2,3,4,5])

        normalize_fcn:     Normalization applied on the results after prediction (default:'min-max')

        shuffle:           Shuffle data (default:False)

        train:             Only train model (default:False)

        predict:           Only predict test set (default:False)

        eval_explanations: Evaluate attentions against spans (default:False)
    '''
    # load mappings:
    label_map = load_mappings(f"data/{dataset_name}/splits/", label_column)

    # load tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for i in iterations:
        print("\n----------------------------------------------------------------------------------------------------")
        print(f" {model_name}: iteration {i:d}")
        print(f" (label = '{label_column}', lr = {str(learning_rate)})")
        print("----------------------------------------------------------------------------------------------------\n")

        # define paths:
        model_dir = f"models/{model_name}/{model_name}-{label_column}-{i:d}"
        result_path = f"results/{model_name}/{model_name}-{label_column}-{i:d}"

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
            trainer = TrainerTransformer(
                num_labels=len(label_map),
                batch_size=batch_size,
                loss_fcn=torch.nn.CrossEntropyLoss(
                    #weight=None,
                    reduction='mean'
                ),
                normalize_fcn=normalize_fcn,
                model_dir=model_dir
            )

            # train model:
            trainer(
                model=model_name,
                data_train=data_train,
                data_valid=data_valid,
                modelargs={
                    'lr':learning_rate,
                    'epochs':epochs,
                    'patience':patience,
                    'shuffle':[shuffle]
                }
            )

            # save trained model:
            # already done during training!
#            trainer.save_model(path=model_dir)

            # save history:
            with open(model_dir + '/hist.json', 'w') as file:
                file.write(str(trainer.train_history))

        if not (train or eval_explanations):
            # load model:
            evaluator = EvaluatorTransformer.load(
                dir=os.path.join(model_dir, 'f1'),
                batch_size=batch_size,
                loss_fcn=torch.nn.CrossEntropyLoss(
                    #weight=None,
                    reduction='mean'
                ),
                num_labels=len(label_map),
                normalize_fcn=normalize_fcn,
                output_attentions=False,
                output_hidden_states=False
            )

            # predict test set:
            results = evaluator.predict(
                data_test,
                output_spans=False,
                output_attentions=False,
                output_hidden_states=False
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
        help='Name of the model to be trained (e.g.: "bert-base-uncased")'
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
    parser.add_argument('--learning_rate',
        metavar='-lr',
        nargs='+',
        type=float,
        default=[5e-5],
        help='Learning rate (Adam): 5e-5, 3e-5, 2e-5 [Devlin et al.]'
    )
    parser.add_argument('--batch_size',
        metavar='-bs',
        type=int,
        default=8,
        help='Batch size: 16, 32 [Devlin et al.]'
    )
    parser.add_argument('--epochs',
        metavar='-e',
        nargs='+',
        type=int,
        default=[4],
        help='Number of epochs: 2, 3, 4 [Devlin et al.]'
    )
    parser.add_argument('--patience',
        metavar='-p',
        nargs='+',
        type=int,
        default=[1],
        help='Patience for early stopping (0 means no early stopping)'
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
    parser.add_argument('--shuffle',
        action='store_true',
        help='Shuffle data'
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