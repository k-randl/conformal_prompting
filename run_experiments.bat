@echo off

set text_name=title
set data_name=incidents
set label_name=%1

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Naive:                                                                                         ::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
python TrainerClassic.py bow-rnd %text_name% %label_name% %data_name%
python TrainerClassic.py bow-sup %text_name% %label_name% %data_name%

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Bag-of-Words:                                                                                  ::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
python TrainerClassic.py bow-knn %text_name% %label_name% %data_name%
python TrainerClassic.py bow-lr  %text_name% %label_name% %data_name%
python TrainerClassic.py bow-svm %text_name% %label_name% %data_name%

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: TF-IDF:                                                                                        ::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
python TrainerClassic.py tfidf-knn %text_name% %label_name% %data_name%
python TrainerClassic.py tfidf-lr  %text_name% %label_name% %data_name%
python TrainerClassic.py tfidf-svm %text_name% %label_name% %data_name%

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Transformers:                                                                                  ::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
python TrainerTransformer.py roberta-base %text_name% %label_name% %data_name% -e 20 -bs 16 -p 5 -nf softmax --shuffle
python TrainerTransformer.py xlm-roberta-base %text_name% %label_name% %data_name% -e 20 -bs 16 -p 5 -nf softmax --shuffle