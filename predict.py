import os
import sys 
import pandas as pd
from resources.evaluator import Evaluator, T_norm
from resources.data_io import ClassificationDataset
from TrainerCICLe import TrainerCICLe
from TrainerClassic import TrainerClassic
from TrainerTransformer import TrainerTransformer

from typing import Iterable

def predict(model_dir:str, text:Iterable[str], normalize_fcn:T_norm=None) -> None:
    '''Predict one or more texts.

    arguments:
        model_dir:      Path to the saved model.

        texts:          Texts to be assessed by the model.

        normalize_fcn:  Normalization applied on the results after prediction (default:None)
    '''

    # load model:
    evaluator = Evaluator.load(dir=model_dir, normalize_fcn=normalize_fcn)

    # create dataframe:
    df = pd.DataFrame(data={'text': text})

    # create dataset:
    data = (ClassificationDataset(data=df, text_column='text', tokenizer=evaluator.tokenizer, add_texts=True), ['input_ids', 'labels', 'texts'])

    # evaluate model:
    outputs = evaluator.predict(data=data, output_spans=False, output_probabilities=False)
    df['predictions'] = outputs['predictions']

    print(df.to_json())

####################################################################################################
# Main Function:                                                                                   #
####################################################################################################

if __name__ == '__main__':
    from argparse import ArgumentParser

    # create parser:
    parser = ArgumentParser(description='Predict one or more texts.')
    parser.add_argument('-nf', '--normalize_fcn',
        metavar='NF',
        type=str,
        default=None,
        help='normalization applied on the results after prediction (default: no normalization)'
    )
    parser.add_argument('model_dir',
        type=str,
        help='path to the saved model'
    )
    parser.add_argument('text',
        nargs='+',
        type=str,
        help='text to be assessed by the model'
    )
    # parse arguments:
    args = parser.parse_args()

    # run main function:
    sys.exit(predict(**dict(args._get_kwargs())))