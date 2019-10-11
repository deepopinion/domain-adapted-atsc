## LM Finetuning

The LM finetuning code is an adaption to a script from the huggingface/pytorch-transformers repository:
* https://github.com/huggingface/pytorch-transformers/blob/v1.0.0/examples/lm_finetuning/finetune_on_pregenerated.py

Prepare the finetuning corpus, here shown for a test corpus "dev_corpus.txt":

    python pregenerate_training_data.py \
    --train_corpus dev_corpus.txt \
    --bert_model bert-base-uncased --do_lower_case \
    --output_dir dev_corpus_prepared/ \
    --epochs_to_generate 2 --max_seq_len 256


Run actual finetuning with:

    python finetune_on_pregenerated.py \
    --pregenerated_data dev_corpus_prepared/ \
    --bert_model bert-base-uncased --do_lower_case \
    --output_dir dev_corpus_finetuned/ \
    --epochs 2 --train_batch_size 16

    
## Downstream Classification

Down-stream task-specific finetuning code is an adaption to this script:
* https://github.com/huggingface/pytorch-transformers/blob/v1.0.0/examples/run_glue.py

In order to train the BERT-ADA Restaurants model on the SemEval 2014 Task 4 Subtask 2
Restaurants dataset, use the command below:

Important: Before running this command download the released finetuned models and
put the into "data/models" folder (see global README of this Repo).
Also the pytorch-transformers and apex python modules need to be installed (apex for mixed precision support).

    
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
