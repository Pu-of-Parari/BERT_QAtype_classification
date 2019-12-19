#!/bin/sh

#set_0
CUDA_VISIBLE_DEVICES=1 python run_classifier_Atype.py --task_name=ATYPE --do_train=true  --do_eval=true --data_dir=./atype_dataset/set_0/  --vocab_file=multi_cased_L-12_H-768_A-12/vocab.txt  --bert_config_file=multi_cased_L-12_H-768_A-12/bert_config.json  --init_checkpoint=multi_cased_L-12_H-768_A-12/bert_model.ckpt  --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5  --num_train_epochs=3.0 --output_dir=./result/atype_v1.4/set_0/ --do_lower_case=false

#set_1
CUDA_VISIBLE_DEVICES=1 python run_classifier_Atype.py --task_name=ATYPE --do_train=true  --do_eval=true --data_dir=./atype_dataset/set_1/  --vocab_file=multi_cased_L-12_H-768_A-12/vocab.txt  --bert_config_file=multi_cased_L-12_H-768_A-12/bert_config.json  --init_checkpoint=multi_cased_L-12_H-768_A-12/bert_model.ckpt  --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5  --num_train_epochs=3.0 --output_dir=./result/atype_v1.4/set_1/ --do_lower_case=false

#set_2
CUDA_VISIBLE_DEVICES=1 python run_classifier_Atype.py --task_name=ATYPE --do_train=true  --do_eval=true --data_dir=./atype_dataset/set_2/  --vocab_file=multi_cased_L-12_H-768_A-12/vocab.txt  --bert_config_file=multi_cased_L-12_H-768_A-12/bert_config.json  --init_checkpoint=multi_cased_L-12_H-768_A-12/bert_model.ckpt  --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5  --num_train_epochs=3.0 --output_dir=./result/atype_v1.4/set_2/ --do_lower_case=false

#set_3
CUDA_VISIBLE_DEVICES=1 python run_classifier_Atype.py --task_name=ATYPE --do_train=true  --do_eval=true --data_dir=./atype_dataset/set_3/  --vocab_file=multi_cased_L-12_H-768_A-12/vocab.txt  --bert_config_file=multi_cased_L-12_H-768_A-12/bert_config.json  --init_checkpoint=multi_cased_L-12_H-768_A-12/bert_model.ckpt  --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5  --num_train_epochs=3.0 --output_dir=./result/atype_v1.4/set_3/ --do_lower_case=false

#set_4
CUDA_VISIBLE_DEVICES=1 python run_classifier_Atype.py --task_name=ATYPE --do_train=true  --do_eval=true --data_dir=./atype_dataset/set_4/  --vocab_file=multi_cased_L-12_H-768_A-12/vocab.txt  --bert_config_file=multi_cased_L-12_H-768_A-12/bert_config.json  --init_checkpoint=multi_cased_L-12_H-768_A-12/bert_model.ckpt  --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5  --num_train_epochs=3.0 --output_dir=./result/atype_v1.4/set_4/ --do_lower_case=false


#set_5
CUDA_VISIBLE_DEVICES=1 python run_classifier_Atype.py --task_name=ATYPE --do_train=true  --do_eval=true --data_dir=./atype_dataset/set_5/  --vocab_file=multi_cased_L-12_H-768_A-12/vocab.txt  --bert_config_file=multi_cased_L-12_H-768_A-12/bert_config.json  --init_checkpoint=multi_cased_L-12_H-768_A-12/bert_model.ckpt  --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5  --num_train_epochs=3.0 --output_dir=./result/atype_v1.4/set_5/ --do_lower_case=false


#set_6
CUDA_VISIBLE_DEVICES=1 python run_classifier_Atype.py --task_name=ATYPE --do_train=true  --do_eval=true --data_dir=./atype_dataset/set_6/  --vocab_file=multi_cased_L-12_H-768_A-12/vocab.txt  --bert_config_file=multi_cased_L-12_H-768_A-12/bert_config.json  --init_checkpoint=multi_cased_L-12_H-768_A-12/bert_model.ckpt  --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5  --num_train_epochs=3.0 --output_dir=./result/atype_v1.4/set_6/ --do_lower_case=false

#set_7
CUDA_VISIBLE_DEVICES=1 python run_classifier_Atype.py --task_name=ATYPE --do_train=true  --do_eval=true --data_dir=./atype_dataset/set_7/  --vocab_file=multi_cased_L-12_H-768_A-12/vocab.txt  --bert_config_file=multi_cased_L-12_H-768_A-12/bert_config.json  --init_checkpoint=multi_cased_L-12_H-768_A-12/bert_model.ckpt  --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5  --num_train_epochs=3.0 --output_dir=./result/atype_v1.4/set_7/ --do_lower_case=false

#set_8
CUDA_VISIBLE_DEVICES=1 python run_classifier_Atype.py --task_name=ATYPE --do_train=true  --do_eval=true --data_dir=./atype_dataset/set_8/  --vocab_file=multi_cased_L-12_H-768_A-12/vocab.txt  --bert_config_file=multi_cased_L-12_H-768_A-12/bert_config.json  --init_checkpoint=multi_cased_L-12_H-768_A-12/bert_model.ckpt  --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5  --num_train_epochs=3.0 --output_dir=./result/atype_v1.4/set_8/ --do_lower_case=false

#set_9
CUDA_VISIBLE_DEVICES=1 python run_classifier_Atype.py --task_name=ATYPE --do_train=true  --do_eval=true --data_dir=./atype_dataset/set_9/  --vocab_file=multi_cased_L-12_H-768_A-12/vocab.txt  --bert_config_file=multi_cased_L-12_H-768_A-12/bert_config.json  --init_checkpoint=multi_cased_L-12_H-768_A-12/bert_model.ckpt  --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5  --num_train_epochs=3.0 --output_dir=./result/atype_v1.4/set_9/ --do_lower_case=false


echo "complete :)"
