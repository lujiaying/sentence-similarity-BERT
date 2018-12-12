#python run_classifier.py --bert_model "bert-base-uncased" --data_dir "/home/jiaying4/projects/glue_data/MRPC" --task_name "mrpc" --output_dir "/home/jiaying4/projects/sentence-similarity-BERT/checkpoints/MRPC" --do_train --do_eval --train_batch_size 64 --eval_batch_size 32 --model_suffix "fist_try_20_epoches" --num_train_epochs 20
#python run_classifier.py --bert_model "bert-base-uncased" --data_dir "/home/jiaying4/projects/glue_data/MRPC" --task_name "mrpc" --output_dir "/home/jiaying4/projects/sentence-similarity-BERT/checkpoints/MRPC" --do_evaluate_sample

#2018-12-10 16:16
#python run_classifier.py --bert_model "bert-base-uncased" --data_dir "./datas/" --task_name "all" --output_dir "/home/jiaying4/projects/sentence-similarity-BERT/checkpoints/all" --do_train --train_batch_size 128 --eval_batch_size 64 --model_suffix "bert-base-uncased" --num_train_epochs 50 --early_training_epoch_threshold 5 --log_step 500
#2018-12-11 12:01
python run_classifier.py --bert_model "bert-base-uncased" --data_dir "./datas/" --task_name "all" --output_dir "/home/jiaying4/projects/sentence-similarity-BERT/checkpoints/all" --do_evaluate_sample --checkpoint_path "./checkpoints/all/all_bert-base-uncased_Epoch7.ckpt"
