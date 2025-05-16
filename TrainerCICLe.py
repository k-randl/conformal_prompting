# imports:
import os
import sys
import pickle

import numpy as np

from tqdm.autonotebook import tqdm

from resources.models import GPT, Llama, Gemma
from resources.embedding import Embedding
from resources.evaluator import Evaluator, EvaluatorConformalSimple, T_norm
from resources.data_io import load_data, load_mappings, T_data

from TrainerClassic import TrainerClassic
from TrainerTransformer import TrainerTransformer

####################################################################################################
# Type hints:                                                                                      #
####################################################################################################

import numpy.typing as npt
from typing import Callable, Optional, Tuple, Iterable, Union

####################################################################################################
# Trainer Class:                                                                                   #
####################################################################################################

class TrainerCICLe():
    def __init__(self, classifier:str, llm:Union[str, Callable[[str,],str]], prompt:str, divider:str='->', embedding:Optional[str]=None, normalize_fcn:T_norm=None, secret:Optional[str]=None) -> None:
        '''Create a new TrainerCICLe object.

        arguments:
            llm:            Name of the LLM (e.g.: "gpt-3.5-turbo-instruct")

            classifier:     Name of the pre-trained base model (e.g.: "models/tfidf-lr/tfidf-lr-hazard-category-0")

            prompt:         The prompt as a Python format string. "{0}" will be replaced with the few-shot samples and "{1}" with the sample to be classified

            divider:        A string dividing samples and labels in the prompt (default:" -> ")

            embedding:      

            normalize_fcn:  Normalization applied on the results after prediction (default:no normalization)

            secret:         An optional secret used to access the LLM (for GPT: openAI API-key; for Llama/Gemma: Huggingface API-key)
        '''

        # load model:
        self._classifier = EvaluatorConformalSimple(Evaluator.load(dir=classifier, normalize_fcn=normalize_fcn))

        # load embedding for similarity computation:
        if embedding is not None:
            print(embedding)
            self._embedding = Embedding.load(embedding)

        elif hasattr(self._classifier._base, 'embedding'):
            self._embedding = self._classifier._base.embedding

        else: raise(ValueError('"embedding" needs to be the path to a valid embedding.'))

        # get llm:
        if isinstance(llm, str):
            if llm.lower().startswith('gpt'):            self._llm = GPT(name=llm, secret=secret)
            elif llm.lower().startswith('meta-llama'):   self._llm = Llama(name=llm, secret=secret)
            elif llm.lower().startswith('google/gemma'): self._llm = Gemma(name=llm, secret=secret)
            else: raise ValueError(llm)

        else: self._llm = llm

        self._prompt = prompt
        self._divider = divider

    @property
    def last_spans(self) -> Union[Iterable[npt.NDArray], npt.NDArray, None]:
        '''`Iterable[NDArray] | NDArray | None`: the spans of the last evaluated dataset as a binary mask (per token).'''
        if self._classifier is not None:
            return self._classifier.last_spans
        else: return None

    @property
    def last_texts(self) -> Union[Iterable[str], None]:
        '''`Iterable[str] | None]`: the texts of the last evaluated dataset.'''
        if self._classifier is not None:
            return self._classifier.last_texts
        else: return None

    @property
    def num_labels(self) -> int:
        '''`int`: the number of unique labels predicted by the model.'''
        if self._classifier is not None:
            return len(self._classifier._labels)
        else: return None

    @property
    def tokenizer(self) -> Callable[[str], Iterable[int]]:
        '''A callable that tokenizes strings to be used by the model.'''
        if self._classifier is not None:
            return self._classifier.tokenizer
        else: return None

    def _enumerate_data(self, data:T_data, desc:str='') -> Iterable[Tuple[str, str, str]]:
        # unpack keys:
        keys = ('texts', 'labels')
        if isinstance(data, tuple):
            data, keys = data

        # iterate through dataset:
        t = ('texts' in keys)
        l = ('labels' in keys)
        s = ('spans' in keys)
        for entry in tqdm(data, desc=desc):
            yield (
                entry['texts'] if t else None,
                entry['labels'].detach().numpy() if l else None,
                entry['spans'].detach().numpy() if s else None
            )

    def _get_samples(self, text:str, classes:Iterable[str], samples_per_class:int=2):
        samples = []

        for l in classes:
            # extract texts of class:
            texts = self._samples[l]

            # get closest sample based on embedding from training data:
            similarity = self._embedding.cosine_similarity(
                [self._embedding.tokenizer(text, return_offsets_mapping=False)['input_ids']] + texts['input_ids']
            )[1:,0]

            for j in np.argsort(similarity)[::-1][:samples_per_class]:
                samples.append((texts['texts'][j], l, similarity[j]))

        # sort samples based on embedding from training data:
        samples.sort(key=lambda e: e[2], reverse=True)

        return samples

    def predict(self, data:T_data, alpha:int=.95, **kwargs) -> Iterable[str]:
        '''Predict some data.

        arguments:
            data:   Data to be predicted.

            alpha:  Conformity parameter.
        '''
        # get most probable classes using conformal prediction:
        result = self._classifier.predict(alpha=alpha, data=data, min_k=1, **kwargs)

        result['candidates'] = result['predictions']
        result['predictions'] = []
        for t, p in tqdm(zip(self._classifier.last_texts, result['candidates']),
                         total=len(self._classifier.last_texts),
                         desc='Predicting texts'):
            if len(p) > 1:
                # build prompt:
                prompt = self._prompt.format(
                    '\n'.join(['"' + x.replace('"',"'") + f'" {self._divider} {y}'
                               for x, y, _ in self._get_samples(t, p['y'])]),
                    '"' + t.replace('"',"'") + f'" {self._divider} '
                )

                # feed through llm:
                result['predictions'].append(self._llm.predict([prompt]))

            else: result['predictions'].append(p['y'][0])

        return result

    @staticmethod
    def load(dir:str, normalize_fcn:T_norm=None, secret:Optional[str]=None, **kwargs) -> 'TrainerCICLe':
        '''Loads a model from disk.

            dir:            Path to the directory that contains the model (e.g.: .\models)

            normalize_fcn:  Normalization function used on the predictions [`"max"` | `"sum"` | `"min-max"` | `"softmax"` | `None`]

            secret:         An optional secret used to access the LLM (for GPT: openAI API-key; for Llama/Gemma: Huggingface API-key)


            returns:        `TrainerClassic` object
        '''
        # load data from file:
        with open(dir + '/model.pickle', 'rb') as file:
            data = pickle.load(file)

        # create evaluator instance:
        trainer =  TrainerCICLe(
            #embedding     = data['embedding_type'].load(os.path.join(dir, 'embedding'), **kwargs),
            #classifier    = data['classifier_type'].load(os.path.join(dir, 'classifier'), **kwargs),
            embedding     = os.path.join(dir, 'embedding'),
            classifier    = os.path.join(dir, 'classifier'),
            llm           = data['llm_name'],
            prompt        = data['prompt'],
            divider       = data['divider'],
            normalize_fcn = normalize_fcn,
            secret        = secret
        )

        # Update samples:
        trainer._samples = data['samples']

        # Update calibration scores:
        trainer._classifier.cal_scores = data['cal_scores']

        return trainer

    def save(self, dir:str, **kwargs) -> None:
        '''Saves a model to disk.

            dir:           Path to the directory that contains the model (e.g.: .\models)
        '''
        # create directory:
        os.makedirs(dir, exist_ok=True)

        # save model:
        with open(dir + '/model.pickle', 'wb') as file:
            pickle.dump({
                'type':             type(self),
                'llm_name':         self._llm.model_name,
                'embedding_type':   type(self._embedding),
                'classifier_type':  type(self._classifier._base),
                'prompt':           self._prompt,
                'divider':          self._divider,
                'samples':          self._samples,
                'cal_scores':       self._classifier.cal_scores
            }, file)

        # save base classifier:
        self._classifier._base.save(os.path.join(dir, 'classifier'), **kwargs)

        # save embedding:
        self._embedding.save(os.path.join(dir, 'embedding'), **kwargs)

    def fit(self, data_samples:T_data, data_calibrate:T_data) -> None:
        '''Calibrate a CICLe model from a previously trained base classifier.

        arguments:
            data_samples:   Pool of few-shot samples.

            data_calibrate: Calibration data for conformal prediction.
        '''

        # calibrate model:
        self._classifier.calibrate(data=data_calibrate)

        # save few-shot samples:
        self._samples = {l:{'texts':[], 'input_ids':[]} for l in self._classifier._labels}
        for text, label, _ in self._enumerate_data(data_samples, desc='Building sample pool'):
            self._samples[self._classifier._labels[int(label)]]['texts'].append(text)
            self._samples[self._classifier._labels[int(label)]]['input_ids'].append(
                self._embedding.tokenizer(text, return_offsets_mapping=False)['input_ids']
            )

####################################################################################################
# Main-Function:                                                                                   #
####################################################################################################

def run(base_name:str, llm_name:str, text_column:str, label_column:str, dataset_name:str,
        divider:           str             = " => ",
        iterations:        Iterable[int]   = range(5),
        normalize_fcn:     T_norm          = None,
        shuffle:           bool            = False,
        train:             bool            = False,
        predict:           bool            = False,
        eval_explanations: bool            = False,
        secret:            Optional[str]   = None
    ) -> None:
    '''Train and/or Evaluate a baseline model.

    arguments:
        llm_name:          Name of the LLM (e.g.: "gpt-3.5-turbo-instruct")

        base_name:         Name of the pre-trained base model (e.g.: "tfidf-lr")

        prompt:            The prompt as a Python format string. "{0}" will be replaced with the few-shot samples and "{1}" with the sample to be classified

        text_column:       Name of the column to be used as the model's input

        label_column:      Name of the column to be used as the label

        dataset_name:      Name of the dataset (e.g.: "incidents")

        divider:           A string dividing samples and labels in the prompt (default:" -> ")

        iterations:        K-Fold iterations (default:[0,1,2,3,4,5])

        normalize_fcn:     Normalization applied on the results after prediction (default:no normalization)

        shuffle:           Shuffle data (default:False)

        train:             Only train model (default:False)

        predict:           Only predict test set (default:False)

        eval_explanations: Evaluate attentions against spans (default:False)

        secret:            An optional secret used to access the LLM (for GPT: openAI API-key; for Llama/Gemma: Huggingface API-key)
    '''
    # load mappings:
    label_map = load_mappings(f"data/{dataset_name}/splits/", label_column)

    for i in iterations:
        print("\n----------------------------------------------------------------------------------------------------")
        print(f" CICLe: iteration {i:d}")
        print(f" (label = '{label_column}', base = '{base_name}', llm = '{llm_name}')")
        print("----------------------------------------------------------------------------------------------------\n")

        # define paths:
        base_dir = f"models/{base_name}/{base_name}-{label_column}-{i:d}"
        model_dir = f"models/cicle/{llm_name}/{base_name}-{label_column}-{i:d}"
        result_dir = f"results/cicle/{llm_name}/{base_name}"

        if not (predict or eval_explanations):
            # build prompt:
            prompt =  f'We are looking for food {label_column.split("-")[0]}s in texts. '
            prompt += 'Here are some labelled examples sorted from most probable to least probable:\n\n'
            prompt += '{0}\n\n'
            prompt += 'Please predict the correct class for the following sample. Answer only with the class label.\n\n'
            prompt += '{1}'

            # create trainer:
            trainer = TrainerCICLe(
                classifier=base_dir,
                llm=llm_name,
                prompt=prompt,
                divider=divider,
                normalize_fcn=normalize_fcn,
                secret=secret
            )
            assert len(label_map) == trainer.num_labels

            # load data:
            data_samples, data_calib, data_test = load_data(
                f"data/{dataset_name}/splits/",
                text_column,
                label_column,
                i,
                trainer.tokenizer,
                add_texts=True
            )

            # calibrate model:
            trainer.fit(
                data_samples=data_samples,
                data_calibrate=data_calib,
            )

            # save calibrated model:
            trainer.save(dir=model_dir)

            # save history:
            #with open(model_dir + '/hist.json', 'w') as f:
            #    json.dump(trainer.train_history, f)

        else:
            # load model:
            trainer = TrainerCICLe.load(
                dir=model_dir,
                num_labels=len(label_map),
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
            # predict test set:
            results = trainer.predict(
                data_test,
                output_spans=False
            )

            # save predictions:
            os.makedirs(result_dir, exist_ok=True)
            with open(f'{result_dir}/CICLe-{label_column}-{i:d}.pickle', 'wb') as f:
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
    parser.add_argument('base_name',
        type=str,
        help='name of the pre-trained base model (e.g.: "tfidf-lr")'
    )
    parser.add_argument('llm_name',
        type=str,
        help='name of the LLM (e.g.: "gpt-3.5-turbo-instruct")'
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
    parser.add_argument('-s', '--secret',
        metavar='S',
        type=str,
        default=None,
        help='An optional secret used to access the LLM (for GPT: openAI API-key; for Llama/Gemma: Huggingface API-key)'
    )

    # parse arguments:
    args = parser.parse_args()

    # run main function:
    sys.exit(run(**dict(args._get_kwargs())))
