import getpass
import numpy as np
from tqdm.autonotebook import tqdm

####################################################################################################
# Type hints:                                                                                      #
####################################################################################################

import abc
import numpy.typing as npt
from typing import Iterable, Optional, Callable

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
        m = np.array([y == c for c in self.classes_], dtype=float)

        if w is not None:
            if len(w) == self._n:   m = np.apply_along_axis(lambda col: col*w, 0, m)
            elif len(w) == len(y):  m = np.apply_along_axis(lambda row: row*w, 1, m)
            else: raise ValueError(f'w must either have one entry per class label ({self._n:d}) or per sample ({len(y):d}), but has length {len(w):d}.')

        self._p = np.mean(m, axis=1)

class RandomModel(NaiveModel):
    def __init__(self, num_classes:int=2) -> None:
        super().__init__(num_classes)

    def predict(self, X:Iterable) -> npt.NDArray:
        return np.random.randint(self._n, size=len(X))
    
    def predict_proba(self, X:Iterable) -> npt.NDArray:
        return np.random.random((len(X), self._n))

class DummyModel(NaiveModel):
    def __init__(self, output:int, num_classes:int=2) -> None:
        super().__init__(num_classes)
        self._output = output

    def predict(self, X:Iterable) -> npt.NDArray:
        return np.ones(len(X), dtype=int) * self._output
    
    def predict_proba(self, X:Iterable) -> npt.NDArray:
        p = np.zeros((len(X), self._n), dtype=float)
        p[:, self._output] = 1.
        return p

####################################################################################################
# LLM Wrapper:                                                                                    #
####################################################################################################

class LLM:
    def __init__(self, name:str, mapping:Callable[[str],str]):
        self._name = name
        self._mapping = mapping

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def classes_(self) -> Iterable[int]:
        raise NotImplementedError()

    def predict(self, X:Iterable[str]) -> Iterable[str]:
        return [self._mapping(x) for x in X]

    def predict_proba(self, X:Iterable) -> npt.NDArray:
        raise NotImplementedError()

    def fit(self, X:Iterable, y:Iterable, w:Optional[Iterable]=None) -> None:
        raise NotImplementedError()

class GPT(LLM):
    def __init__(self, name:str='gpt-3.5-turbo-instruct', temperature:float=0.) -> None:
        # Login to OpenAI and create a client:
        from openai import OpenAI
        self._client = OpenAI(api_key=getpass.getpass('Enter your OpenAI API-key:'))

        # Initialize base class:
        super().__init__(name,
            mapping=lambda x: self._client.completions.create(
                model=name,
                prompt=x,
                temperature=temperature
            ).choices[0].text.strip()
        )

class Gemma(LLM):
    def __init__(self, name:str='google/gemma-1.1-7b-it', temperature:Optional[float]=None, top_p:Optional[float]=None) -> None:
        # Login to huggingface:
        from huggingface_hub import login
        login(getpass.getpass('Enter your huggingface API-key:'))

        # Create a pipeline:
        import torch
        from transformers import pipeline
        self._pipeline = pipeline("text-generation",
            model=name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto"
        )

        # Initialize base class:
        super().__init__(name,
            mapping=lambda x: self._pipeline(
                [{'role':'user', 'content':x}],
                max_new_tokens=32,
                do_sample=(top_p is not None) or (temperature is not None),
                temperature=temperature,
                top_p=top_p
            )[0]["generated_text"][-1]["content"]
        )

class Llama(LLM):
    def __init__(self, name:str='meta-llama/Meta-Llama-3.1-8B-Instruct', temperature:Optional[float]=None, top_p:Optional[float]=None) -> None:
        # Login to huggingface:
        from huggingface_hub import login
        login(getpass.getpass('Enter your huggingface API-key:'))

        # Create a pipeline:
        import torch
        from transformers import pipeline
        self._pipeline = pipeline("text-generation",
            model=name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto"
        )

        # Get special tokens:
        bos_token_id = self._pipeline.tokenizer.convert_tokens_to_ids('<|begin_of_text|>')
        eos_token_id = self._pipeline.tokenizer.convert_tokens_to_ids('<|eot_id|>')
        pad_token_id = self._pipeline.tokenizer.convert_tokens_to_ids('<|eot_id|>')

        # Initialize base class:
        super().__init__(name,
            mapping=lambda x: self._pipeline(
                [{'role':'user', 'content':x}],
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                max_new_tokens=32,
                do_sample=(top_p is not None) or (temperature is not None),
                temperature=temperature,
                top_p=top_p
            )[0]["generated_text"][-1]["content"]
        )