python model.py \
  --data_dir=data \
  --model_dir=Res \
  --bert_dir=resources/chinese_L-12_H-768_A-12 \
  --max_seq_length=512 \
  --mode=train_test \
  --batch_size=4 \
  --epochs=3 \
  --attention_type=add \
  --learning_rate=5e-5
