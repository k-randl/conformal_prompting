# imports:
import os
import sys
import json
import pickle
import shutil

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, ConstantLR, LinearLR
from transformers import PreTrainedTokenizerBase, AutoTokenizer, AutoModelForSequenceClassification, PreTrainedModel

import numpy as np
from numpy.lib import recfunctions as rfn

from tqdm import trange
from tqdm.autonotebook import tqdm

from resources.evaluator import Trainer, T_norm
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
from typing import Optional, Tuple, Iterable, Union, Generator, List, Dict, Any

####################################################################################################
# Trainer Class:                                                                                   #
####################################################################################################

class TrainerTransformer(Trainer):
    def __init__(self, model:str, labels:List[str], batch_size:int, normalize_fcn:T_norm=None, loss_fcn:Optional[nn.Module]=None, **kwargs) -> None:
        super().__init__(labels=labels, normalize_fcn=normalize_fcn)
        self._batch_size = batch_size
        self._loss_fcn   = loss_fcn
        self._tokenizer  = AutoTokenizer.from_pretrained(model)
        self._model      = model

    def _init_model(self, model:str, **kwargs):
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model,
            num_labels=self.num_labels,
            **kwargs
        )
        self._model = nn.DataParallel(self._model)
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
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

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
    def load(dir:str, batch_size:int=None, normalize_fcn:T_norm=None, loss_fcn:Optional[nn.Module]=None, **kwargs) -> 'TrainerTransformer':
        '''Loads a model from disk.

            dir:            Path to the directory that contains the model (e.g.: .\models)

            batch_size:     Batch size: 16, 32 [Devlin et al.]

            normalize_fcn:  Normalization function used on the predictions [`"max"` | `"sum"` | `"min-max"` | `"softmax"` | `None`]


            returns:        An instance of `TrainerTransformerTfIdf` with the model.
        '''
        # load data from file:
        with open(dir + '/model.pickle', 'rb') as file:
            data = pickle.load(file)

        # create evaluator instance:
        trainer = TrainerTransformer(
            model=dir,
            labels=data['labels'],
            batch_size=batch_size,
            normalize_fcn=normalize_fcn,
            loss_fcn=loss_fcn,
            **kwargs
        )

        return trainer

    def save(self, dir:str) -> None:
        '''Saves a model to disk.

            dir:           Path to the directory that contains the model (e.g.: .\models)
        '''
        # make sure that model is set:
        assert not (self._model is None)

        # create folder if necessary:
        os.makedirs(dir, exist_ok=True)

        # save metadata:
        with open(dir + '/model.pickle', 'wb') as file:
            pickle.dump({
                'type':           type(self),
                'labels':         self._labels
            }, file)

        # save tokenizer:
        self._tokenizer.save_pretrained(dir)

        # get model and state dict to be saved:
        model = self._model.module if hasattr(self._model, 'module') else self._model
        state_dict = model.state_dict()

        # if model is a PreTrainedModel-instance use transformers save format:
        if issubclass(type(model), PreTrainedModel):
            model.save_pretrained(dir, state_dict=state_dict)

        # otherwise save in torch style:
        else:
            torch.save(state_dict, f'{dir}/{type(model).__name__}.pt')

    def predict(self, data:T_data, output_probabilities:bool=True, output_structured:bool=True, output_spans:bool=False, output_attentions:bool=False, output_hidden_states:bool=False) -> Dict[str, any]:
        assert not (self._model is None)
        self._model.eval()

        eval_loss = 0
        n_steps = 0
        logits, labels = [], []
        spans, attentions, hidden_states = [], [], []

        for labels_tensor, weights_tensor, spans_tensor, kwargs in self._enumerate_data(data, desc="Evaluation"):
            # if no loss fcn specified: use internal loss fcn of model
            if self._loss_fcn is None: kwargs['labels'] = labels_tensor

            # without gradient calculation:
            with torch.no_grad():
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

                # add predictions and labels to lists:
                logits.extend(list(model_out.logits.to('cpu').numpy()))
                labels.extend(list(labels_tensor.to('cpu').numpy()))

                # update variables for mean loss calculation:
                if not (self._loss_fcn is None):
                    loss = self._loss_fcn(
                        model_out.logits.type(torch.float),
                        nn.functional.one_hot(labels_tensor, num_classes=self.num_labels).type(torch.float)
                    )
                    eval_loss += loss.item()
                    n_steps += 1

                elif hasattr(model_out, 'loss'):
                    loss = model_out.loss
                    eval_loss += loss.item()
                    n_steps += 1

        # create and return output dictionary:
        result = {
            'labels':self.id2label(labels),
            'predictions':self.id2label(np.argmax(logits, axis=-1))
        }
        if output_probabilities:    
            result['probabilities'] = np.apply_along_axis(self._normalize_fcn, 1, logits)

            if output_structured:   
                result['probabilities'] = rfn.unstructured_to_structured(
                    arr=result['probabilities'],
                    dtype=np.dtype([(label, 'f4') for label in self._labels])
                )

        if output_spans:            result['spans'] = spans
        if output_attentions:       result['attentions'] = attentions
        if output_hidden_states:    result['hidden_states'] = hidden_states
        if n_steps > 0:             result['loss'] = eval_loss/n_steps
        return result

    def fit(self, data_train:T_data, data_valid:T_data, max_grad_norm:float=1., model_dir:str='./model', modelargs: Dict[str, Iterable[Any]]={}) -> None:
        assert isinstance(self._model, str)
        pretrained_model_name = self._model
        
        model_dir = os.path.abspath(model_dir)
        tmp_dir   = os.path.join(model_dir, '.tmp')

        best_f1   = float('-inf')
        best_loss = float('inf')

        self._train_history = []
        for kwargs in ParameterGrid(modelargs):
            # create new model: 
            self._init_model(pretrained_model_name,
                output_attentions=False,
                output_hidden_states=False
            )

            # train model:
            f1, loss, loss_train, loss_valid = self.train(
                data_train,
                data_valid,
                max_grad_norm=max_grad_norm,
                checkpoint_dir=tmp_dir,
                **kwargs
            ) 

            #save score to history:
            hist_item = kwargs.copy()
            hist_item['max_f1'] = f1
            hist_item['min_loss'] = loss
            hist_item['loss_train'] = loss_train.tolist()
            hist_item['loss_valid'] = loss_valid.tolist()
            self._train_history.append(hist_item)

            if f1 > best_f1:
                print(f'New best f1: {f1:.2f} (args={str(kwargs)})')
                best_f1 = f1
                shutil.copytree(
                    os.path.join(tmp_dir, 'f1'),
                    os.path.join(model_dir, 'f1'),
                    dirs_exist_ok=True
                )

            if loss < best_loss:
                print(f'New best loss: {loss:.2f} (args={str(kwargs)})')
                best_loss = loss
                shutil.copytree(
                    os.path.join(tmp_dir, 'loss'),
                    os.path.join(model_dir, 'loss'),
                    dirs_exist_ok=True
                )

        # clean up:
        shutil.rmtree(tmp_dir, ignore_errors=True)

        # load best:
        self._init_model(os.path.join(model_dir, 'f1'),
            output_attentions=False,
            output_hidden_states=False
        )

    def train(self, data_train:T_data, data_valid:T_data, lr:float=5e-5, epochs:int=1, patience:int=0, max_grad_norm:float=1., pretrain:bool=False, use_f1:bool=True, shuffle:bool=False, checkpoint_dir:str='./checkpoint') -> Tuple[float, float, npt.NDArray, npt.NDArray]:
        '''Model training for several epochs using early stopping.'''

        # create new optimizer and schedule:
        optimizer = AdamW(self._model.parameters(), lr=lr)

        epochs_const = int(0.1 * epochs)
        s_const  = ConstantLR(optimizer, factor=1., total_iters=epochs_const)
        s_linear = LinearLR(optimizer, start_factor=1., end_factor=0.01, total_iters=epochs-epochs_const)
        scheduler = SequentialLR(optimizer,
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
            train_out = self.epoch(data_train, optimizer=optimizer, scheduler=scheduler, max_grad_norm=max_grad_norm, shuffle=shuffle)
            valid_out = self.predict(data_valid, output_probabilities=False)
            loss_train[i] = train_out['loss']
            loss_valid[i] = valid_out['loss']

            # print metrics:
            print(f"\nTraining:")
            print(f"  Loss:      {loss_train[i]:4.2f}")

            print(f"\nValidation:")
            print(f"  Loss:      {loss_valid[i]:4.2f}")

            if use_f1:
                y_true = valid_out['labels']
                y_pred = valid_out['predictions']
                f1_valid[i] = f1_score(y_true, y_pred, average='macro')
                print(f"  F1:        {f1_valid[i]:4.2f}")
                print(f"  Precision: {precision_score(y_true, y_pred, average='macro'):4.2f}")
                print(f"  Recall:    {recall_score(y_true, y_pred, average='macro'):4.2f}\n")

            # deactivate pretraining before saving:
            if pretrain: self._model.reset_pretrain()

            # save models in first round:
            if i == 0:
                if use_f1: self.save(os.path.join(checkpoint_dir, 'f1'))
                self.save(os.path.join(checkpoint_dir, 'loss'))

                # reactivate pretraining after saving:
                if pretrain: self._model.set_pretrain()
                continue

            # save models with best f1:
            if not best_f1_found:
                i_max_f1 = np.argmax(f1_valid[:i])
                if f1_valid[i] > f1_valid[i_max_f1]:
                    self.save(os.path.join(checkpoint_dir, 'f1'))
                elif i_max_f1 <= i - patience:
                    print(f"Early stopping for max. F1 initiated.\nBest epoch: {i_max_f1:d}\n")
                    best_f1_found = True

            # save models with best loss:
            if not best_loss_found:
                i_min_loss = np.argmin(loss_valid[:i])
                if loss_valid[i] < loss_valid[i_min_loss]:
                    self.save(os.path.join(checkpoint_dir, 'loss'))
                elif i_min_loss <= i - patience:
                    print(f"Early stopping for min. loss initiated.\nBest epoch: {i_min_loss:d}\n")
                    best_loss_found = True

            # stop training if done:
            if best_loss_found and best_f1_found: break

            # reactivate pretraining after saving:
            if pretrain: self._model.set_pretrain()

        if use_f1:  return np.nanmax(f1_valid), np.nanmin(loss_valid), loss_train, loss_valid
        else:       return float('-inf'), np.nanmin(loss_valid), loss_train, loss_valid

    def epoch(self, data:T_data, max_grad_norm:float=1., output_spans:bool=False, output_attentions:bool=False, output_hidden_states:bool=False, shuffle:bool=False, optimizer:Optional[nn.Module]=None, scheduler:Optional[nn.Module]=None) -> Dict[str, any]:
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
                loss = self._loss_fcn(
                    model_out.logits.type(torch.float),
                    nn.functional.one_hot(labels_tensor, num_classes=self.num_labels).type(torch.float)
                )

            elif hasattr(model_out, 'loss'):
                loss = model_out.loss

            else: raise AttributeError("No loss function defined.")

            # update variables for mean loss calculation:
            train_loss += loss.item()
            n_steps += 1

            # backpropagate loss:
            loss.backward()

            # clip grads to avoid exploding gradients:
            nn.utils.clip_grad_norm_(self._model.parameters(), max_grad_norm)

            # step optimization:
            if optimizer != None:
                optimizer.step()
                optimizer.zero_grad()

        # step scheduler:
        if scheduler != None:
            scheduler.step()

        # create and return output dictionary:
        result = {}
        if output_spans:            result['spans'] = spans
        if output_attentions:       result['attentions'] = attentions
        if output_hidden_states:    result['hidden_states'] = hidden_states
        if n_steps > 0:             result['loss'] = train_loss/n_steps
        return result

####################################################################################################
# Main-Function:                                                                                   #
####################################################################################################

def run(model_name:str, text_column:str, label_column:str, dataset_name:str,
        learning_rate:     Iterable[float] = [5e-5],
        batch_size:        int             = 8,
        epochs:            Iterable[int]   = [4],
        patience:          Iterable[int]   = [1],
        iterations:        Iterable[int]   = range(5),
        normalize_fcn:     T_norm          = None,
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

        normalize_fcn:     Normalization applied on the results after prediction (default:no normalization)

        shuffle:           Shuffle data (default:False)

        train:             Only train model (default:False)

        predict:           Only predict test set (default:False)

        eval_explanations: Evaluate attentions against spans (default:False)
    '''
    # load mappings:
    label_map = load_mappings(f"data/{dataset_name}/splits/", label_column)

    for i in iterations:
        print("\n----------------------------------------------------------------------------------------------------")
        print(f" {model_name}: iteration {i:d}")
        print(f" (label = '{label_column}', lr = {str(learning_rate)})")
        print("----------------------------------------------------------------------------------------------------\n")

        # define paths:
        model_dir = f"models/{model_name}/{model_name}-{label_column}-{i:d}"
        result_dir = f"results/{model_name}"

        if not (predict or eval_explanations):
            # create trainer:
            trainer = TrainerTransformer(
                model=model_name,
                labels=label_map,
                batch_size=batch_size,
                loss_fcn=torch.nn.CrossEntropyLoss(
                    #weight=None,
                    reduction='mean'
                ),
                normalize_fcn=normalize_fcn
            )

            # load data:
            data_train, data_valid, data_test = load_data(
                f"data/{dataset_name}/splits/",
                text_column,
                label_column,
                i,
                trainer.tokenizer
            )

            # train model:
            trainer.fit(
                data_train=data_train,
                data_valid=data_valid,
                model_dir=model_dir,
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
            with open(model_dir + '/hist.json', 'w') as f:
                json.dump(trainer.train_history, f)

        else:
            # load model:
            trainer = TrainerTransformer.load(
                dir=os.path.join(model_dir, 'loss'),
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

            # load data:
            _, _, data_test = load_data(
                f"data/{dataset_name}/splits/",
                text_column,
                label_column,
                i,
                trainer.tokenizer
            )

        if not (train or eval_explanations):
            # predict test set:
            results = trainer.predict(
                data_test,
                output_spans=False,
                output_attentions=False,
                output_hidden_states=False
            )

            # save predictions:
            os.makedirs(result_dir, exist_ok=True)
            with open(f'{result_dir}/{model_name}-{label_column}-{i:d}.pickle', 'wb') as f:
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
        help='name of the model to be trained (e.g.: "bert-base-uncased")'
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
    parser.add_argument('-lr', '--learning_rate',
        metavar='LR',
        nargs='+',
        type=float,
        default=[5e-5],
        help='learning rate (Adam): 5e-5, 3e-5, 2e-5 [Devlin et al.]'
    )
    parser.add_argument('-bs', '--batch_size',
        metavar='BS',
        type=int,
        default=8,
        help='batch size: 16, 32 [Devlin et al.]'
    )
    parser.add_argument('-e', '--epochs',
        metavar='E',
        nargs='+',
        type=int,
        default=[4],
        help='number of epochs: 2, 3, 4 [Devlin et al.]'
    )
    parser.add_argument('-p', '--patience',
        metavar='P',
        nargs='+',
        type=int,
        default=[1],
        help='patience for early stopping (0 means no early stopping)'
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
    parser.add_argument('--shuffle',
        action='store_true',
        help='shuffle data'
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