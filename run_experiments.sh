#!/bin/sh
text_name=title
data_name=incidents

####################################################################################################
# Naive:                                                                                           #
####################################################################################################
python TrainerClassic.py bow-rnd $text_name $1 $data_name -nf 'softmax' --shuffle
python TrainerClassic.py bow-sup $text_name $1 $data_name -nf 'softmax' --shuffle

####################################################################################################
# Bag-of-Words:                                                                                    #
####################################################################################################
python TrainerClassic.py bow-knn $text_name $1 $data_name -nf 'softmax' --shuffle
python TrainerClassic.py bow-lr  $text_name $1 $data_name -nf 'softmax' --shuffle
python TrainerClassic.py bow-svm $text_name $1 $data_name -nf 'softmax' --shuffle

####################################################################################################
# TF-IDF:                                                                                          #
####################################################################################################
python TrainerClassic.py tfidf-knn $text_name $1 $data_name -nf 'softmax' --shuffle
python TrainerClassic.py tfidf-lr  $text_name $1 $data_name -nf 'softmax' --shuffle
python TrainerClassic.py tfidf-svm $text_name $1 $data_name -nf 'softmax' --shuffle

####################################################################################################
# Transformers:                                                                                    #
####################################################################################################
python TrainerTransformer.py roberta-base $text_name $1 $data_name -e 20 -bs 16 -p 5 -nf 'softmax' --shuffle
python TrainerTransformer.py xlm-roberta-base $text_name $1 $data_name -e 20 -bs 16 -p 5 -nf 'softmax' --shuffle