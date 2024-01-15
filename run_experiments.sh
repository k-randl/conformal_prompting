#!/bin/sh
text_name=title
data_name=incidents
label_name=$1
shift

args=""
for arg in $@
do args="${args} ${arg}"
done

####################################################################################################
# Naive:                                                                                           #
####################################################################################################
python TrainerClassic.py bow-rnd $text_name $label_name $data_name $args
python TrainerClassic.py bow-sup $text_name $label_name $data_name $args

####################################################################################################
# Bag-of-Words:                                                                                    #
####################################################################################################
python TrainerClassic.py bow-knn $text_name $label_name $data_name $args
python TrainerClassic.py bow-lr  $text_name $label_name $data_name $args
python TrainerClassic.py bow-svm $text_name $label_name $data_name $args

####################################################################################################
# TF-IDF:                                                                                          #
####################################################################################################
python TrainerClassic.py tfidf-knn $text_name $label_name $data_name $args
python TrainerClassic.py tfidf-lr  $text_name $label_name $data_name $args
python TrainerClassic.py tfidf-svm $text_name $label_name $data_name $args

####################################################################################################
# Transformers:                                                                                    #
####################################################################################################
python TrainerTransformer.py roberta-base $text_name $label_name $data_name -e 20 -bs 16 -p 5 -nf softmax --shuffle $args
python TrainerTransformer.py xlm-roberta-base $text_name $label_name $data_name -e 20 -bs 16 -p 5 -nf softmax --shuffle $args