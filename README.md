# FritzBot_github
 FritzBot data set and backend

The code was created based on https://github.com/macanv/BERT-BiLSTM-CRF-NER

## How to train

#### 1. Download BERT pretrain model :  

 ```
 mkdir checkpoint && cd checkpoint && 
 wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip 
 ```

#### 2. create output dir

create output path in project path:

```angular2html
mkdir output
```

#### 3. Train model

Then run

```
  python3 run.py   \
                  --task_name="NER"  \ 
                  --do_train=True   \
                  --do_eval=True   \
                  --do_predict=True	\
                  --data_dir=data/student_submissions_rephased   \
                  --vocab_file=checkpoint/uncased_L-12_H-768_A-12/vocab.txt  \ 
                  --bert_config_file=checkpoint/uncased_L-12_H-768_A-12/bert_config.json \  
                  --init_checkpoint=checkpoint/uncased_L-12_H-768_A-12/bert_model.ckpt   \
                  --max_seq_length=128   \
                  --train_batch_size=32   \
                  --learning_rate=2e-5   \
                  --num_train_epochs=128.0   \
                  --output_dir=./output/result_dir/ 	\
                  --num_layers=4	
```

## Predict

Replace [model_dir] and [bert_dir] with your own in script terminal_predict_fritzbot.py, and run it. 

```
python3 terminal_predict_fritzbot.py
```

