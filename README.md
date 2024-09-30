# conformal_prompting
Implementation of the paper "[CICLe: Conformal In-Context Learning for Largescale Multi-Class Food Risk Classification](https://doi.org/10.18653/v1/2024.findings-acl.459)"

**Abstract:** *Contaminated or adulterated food poses a substantial risk to human health. Given sets of labeled web texts for training, Machine Learning and Natural Language Processing can be applied to automatically detect such risks given the necessary labeled training data.
We publish a dataset of 7,546 short texts describing public food recall announcements. Each text is manually labeled, on two granularity levels (coarse and fine), for food products and hazards that the recall corresponds to. 
We describe the dataset and benchmark naive, traditional, and Transformer models. Based on our analysis, Logistic Regression based on a tf-idf representation outperforms RoBERTa and XLM-R on classes with low support.
Finally, we discuss different prompting strategies and present an LLM-in-the-loop framework, based on Conformal Prediction, which boosts the performance of the base classifier while reducing power consumption compared to normal prompting.*

## Usage

If you only want to find out how the method works, check out our [demo](https://github.com/k-randl/conformal_prompting/tree/main/demo/cicle_demo.ipynb).

Otherwise, start with installing the Python modules specified in [requirements.txt](https://github.com/k-randl/conformal_prompting/tree/main/requirements.txt). This can be done by executing the following command in the cloned directory:
```
pip install -r requirements.txt
```
Next, download the dataset by running:
```
python get_data.py
```
Afterwards, you can train and evaluate the baseline models using the following commands:

### Classic ML:
```
TrainerClassic.py [-h] [-i I [I ...]] [-nf NF]
                  [--pca] [--train] [--predict] [--eval_explanations]
                  model_name text_column label_column dataset_name
```

*Train and/or Evaluate a baseline model based on [scikit-learn](https://scikit-learn.org/stable/).*

**Positional Arguments:**
```
  model_name            name of the model to be trained (e.g.: "bow-svm")
  text_column           name of the column to be used as the model's input
  label_column          name of the column to be used as the label
  dataset_name          name of the dataset (e.g.: "incidents")
```

**Options:**
```
  -h, --help            show this help message and exit
  -i I [I ...], --iterations I [I ...]
                        k-fold iterations
  -nf NF, --normalize_fcn NF
                        normalization applied on the results after prediction (default: no normalization)
  --pca                 use principal component analysis for reducing the embedding dimensions
  --train               only train model
  --predict             only predict test set
  --eval_explanations   evaluate attentions against spans
```

### Transformers:
```
TrainerTransformer.py [-h] [-lr LR [LR ...]] [-bs BS] [-e E [E ...]] [-p P [P ...]] [-i I [I ...]] [-nf NF]
                      [--shuffle] [--train] [--predict] [--eval_explanations]
                      model_name text_column label_column dataset_name
```

*Train and/or Evaluate a baseline model from [huggingface.co](https://huggingface.co/).*

**Positional arguments:**
```
  model_name            name of the model to be trained (e.g.: "bert-base-uncased")
  text_column           name of the column to be used as the model's input
  label_column          name of the column to be used as the label
  dataset_name          name of the dataset (e.g.: "incidents")
```

**Options:**
```
  -h, --help            show this help message and exit
  -lr LR [LR ...], --learning_rate LR [LR ...]
                        learning rate (Adam): 5e-5, 3e-5, 2e-5 [Devlin et al.]
  -bs BS, --batch_size BS
                        batch size: 16, 32 [Devlin et al.]
  -e E [E ...], --epochs E [E ...]
                        number of epochs: 2, 3, 4 [Devlin et al.]
  -p P [P ...], --patience P [P ...]
                        patience for early stopping (0 means no early stopping)
  -i I [I ...], --iterations I [I ...]
                        k-fold iterations
  -nf NF, --normalize_fcn NF
                        normalization applied on the results after prediction (default: no normalization)
  --shuffle             shuffle data
  --train               only train model
  --predict             only predict test set
  --eval_explanations   evaluate attentions against spans
```

### CICLe:
After training and saving at least one of the above models you can use them with CICLe:

```
TrainerCICLe.py [-h] [-nf NF] [--shuffle] [--train] [--predict] [--eval_explanations]
                base_name llm_name text_column label_column dataset_name
```

*Train and/or Evaluate a CICLe model.*

**Positional Arguments:**
```
  base_name             name of the pre-trained base model (e.g.: "tfidf-lr")
  llm_name              name of the LLM (e.g.: "gpt-3.5-turbo-instruct")
  text_column           name of the column to be used as the model's input
  label_column          name of the column to be used as the label
  dataset_name          name of the dataset (e.g.: "incidents")
```

**Options:**
```
  -h, --help            show this help message and exit
  -nf NF, --normalize_fcn NF
                        normalization applied on the results after prediction (default: no normalization)
  --shuffle             shuffle data
  --train               only train model
  --predict             only predict test set
  --eval_explanations   evaluate attentions against spans
```

### Simply predicting texts using any pre-trained model (*Classic or Transformer*):
```
predict.py [-h] [-nf NF] model_dir text [text ...]
```

*Predict one or more texts. Results are printed to std-out as a json-string.*

**Positional arguments:**
```
  model_dir             path to the saved model
  text                  text to be assessed by the model
```

**Options:**
```
  -h, --help            show this help message and exit
  -nf NF, --normalize_fcn NF
                        normalization applied on the results after prediction (default: no normalization)
```

## Citing
If you use our work please cite the following article:
```
@inproceedings{randl-etal-2024-cicle,
    title = "{CICL}e: Conformal In-Context Learning for Largescale Multi-Class Food Risk Classification",
    author = "Randl, Korbinian  and Pavlopoulos, John  and Henriksson, Aron  and Lindgren, Tony",
    editor = "Ku, Lun-Wei  and Martins, Andre  and Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.459",
    doi = "10.18653/v1/2024.findings-acl.459",
    pages = "7695--7715",
}
```