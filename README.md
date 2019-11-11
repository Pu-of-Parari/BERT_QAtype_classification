## BERT Answer Type Classification
sentence classification for answer type(sent, phrase, word, num, other)

### Implemention Environment
```
tensorflow-gpu==1.14.0
```

### Use data format
These to some directory $DATASET_DIR.
- `train.tsv`
- `dev.tsv`
- `test.tsv`

This is example of data format.
```
index	sentence	a_type
1212	直接 会っ て お願い する	sent
1213	結婚式 の 6カ月 前 ～ 1年前	sent
```

### Useage
This repository's scripts based on https://github.com/google-research/bert .
Use them on BERT dir.

- BERT fine-tuning

```
python run_classifier_Atype.py \
--task_name=ATYPE --do_train=true \
--do_eval=true --data_dir=./$DATASET_DIR \
--vocab_file=multi_cased_L-12_H-768_A-12/vocab.txt \
--bert_config_file=multi_cased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=multi_cased_L-12_H-768_A-12/bert_model.ckpt \
--max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 \
--num_train_epochs=3.0 --output_dir=./result/atype/ --do_lower_case=false
```

- Predict for test dataset

By runnning this command, you can obtain `test_results.tsv` on your output_dir.

```
python run_classifier_Atype.py \
--task_name=ATYPE --do_predict=true --data_dir=./aTypeClassifier/ \
--vocab_file=./multi_cased_L-12_H-768_A-12/vocab.txt \
--bert_config_file=./multi_cased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=./result/atype/ --max_seq_length=128 \
--output_dir=./result/atype/atype_output --do_lower_case=false
```


- evaluation 

This code outputs accuracy and confusion map of test prediction.
Before running it, you need to add the header to `test_results.tsv`
```
index	word	sent	phrase	num	other
```
and add index for each samples.

You can evaluate as follows,

```
python test_eval.py
```
