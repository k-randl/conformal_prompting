# conformal_prompting

## Usage

**Bag of Words:**
```
TrainerBOW.py [-h] [--iterations -i [-i ...]] [--normalize_fcn -nf] [--threads -t] [--pca] [--train] [--predict] [--eval_explanations] model_name text_column label_column dataset_name

Train and/or Evaluate a baseline model.

positional arguments:
  model_name            Name of the model to be trained (e.g.: "svm")
  text_column           Name of the column to be used as the model's input
  label_column          Name of the column to be used as the label
  dataset_name          Name of the dataset (e.g.: "incidents")

options:
  -h, --help            show this help message and exit
  --iterations -i [-i ...]
                        K-Fold iterations
  --normalize_fcn -nf   Normalization applied on the results after prediction (default:'min-max')
  --threads -t          Number of threads for parallel processsing
  --pca                 Use Principal Component Analysis for reducing the embeding dimensions
  --train               Only train model
  --predict             Only predict test set
  --eval_explanations   Evaluate attentions against spans
```

**TF-IDF:**
```
TrainerTfIdf.py [-h] [--iterations -i [-i ...]] [--normalize_fcn -nf] [--threads -t] [--pca] [--train] [--predict] [--eval_explanations] model_name text_column label_column dataset_name

Train and/or Evaluate a baseline model.

positional arguments:
  model_name            Name of the model to be trained (e.g.: "svm")
  text_column           Name of the column to be used as the model's input
  label_column          Name of the column to be used as the label
  dataset_name          Name of the dataset (e.g.: "incidents")

options:
  -h, --help            show this help message and exit
  --iterations -i [-i ...]
                        K-Fold iterations
  --normalize_fcn -nf   Normalization applied on the results after prediction (default:'min-max')
  --threads -t          Number of threads for parallel processsing
  --pca                 Use Principal Component Analysis for reducing the embeding dimensions
  --train               Only train model
  --predict             Only predict test set
  --eval_explanations   Evaluate attentions against spans
```

**Transformers:**
```
TrainerTransformer.py [-h] [--learning_rate -lr [-lr ...]] [--batch_size -bs] [--epochs -e [-e ...]] [--patience -p [-p ...]] [--iterations -i [-i ...]] [--normalize_fcn -nf] [--shuffle]
                             [--train] [--predict] [--eval_explanations]
                             model_name text_column label_column dataset_name

Train and/or Evaluate a baseline model.

positional arguments:
  model_name            Name of the model to be trained (e.g.: "bert-base-uncased")
  text_column           Name of the column to be used as the model's input
  label_column          Name of the column to be used as the label
  dataset_name          Name of the dataset (e.g.: "incidents")

options:
  -h, --help            show this help message and exit
  --learning_rate -lr [-lr ...]
                        Learning rate (Adam): 5e-5, 3e-5, 2e-5 [Devlin et al.]
  --batch_size -bs      Batch size: 16, 32 [Devlin et al.]
  --epochs -e [-e ...]  Number of epochs: 2, 3, 4 [Devlin et al.]
  --patience -p [-p ...]
                        Patience for early stopping (0 means no early stopping)
  --iterations -i [-i ...]
                        K-Fold iterations
  --normalize_fcn -nf   Normalization applied on the results after prediction (default:'min-max')
  --shuffle             Shuffle data
  --train               Only train model
  --predict             Only predict test set
  --eval_explanations   Evaluate attentions against spans
```

