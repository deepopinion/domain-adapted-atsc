## Downstream Classification

In order to train the BERT-ADA Restaurants model on the SemEval 2014 Task 4 Subtask 2
Restaurants dataset, use the following command.

    python run_glue.py \ 
    --model_type bert \
    --model_name_or_path ../data/models/restaurants_10mio_ep3 \
    --do_train --evaluate_during_training --do_eval \
    --logging_steps 100 --save_steps 1200 --task_name=semeval2014-atsc \
    --seed 42 --do_lower_case \
    --data_dir=../data/transformed/restaurants_noconfl \
    --output_dir=../data/models/semeval2014-atsc-bert-ada-restaurants-restaurants \
    --max_seq_length=128 --learning_rate 3e-5 --per_gpu_eval_batch_size=32 --per_gpu_train_batch_size=32 \
    --gradient_accumulation_steps=1 --max_steps=800 --overwrite_output_dir --overwrite_cache --warmup_steps=120 --fp16

* Important: Before running this command download the released finetuned models and
put the into "data/models" folder (see global README of this Repo).
Also the pytorch-transformers and apex python modules need to be installed (apex for mixed precision support).

* Note: This code has not been fully tested as it was produced by a quick cleanup of the original code.
If you find any errors, please create an github issue.
