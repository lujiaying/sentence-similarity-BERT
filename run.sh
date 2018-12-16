#python run_classifier.py --bert_model "bert-base-uncased" --data_dir "/home/jiaying4/projects/glue_data/MRPC" --task_name "mrpc" --output_dir "/home/jiaying4/projects/sentence-similarity-BERT/checkpoints/MRPC" --do_train --do_eval --train_batch_size 64 --eval_batch_size 32 --model_suffix "fist_try_20_epoches" --num_train_epochs 20
#python run_classifier.py --bert_model "bert-base-uncased" --data_dir "/home/jiaying4/projects/glue_data/MRPC" --task_name "mrpc" --output_dir "/home/jiaying4/projects/sentence-similarity-BERT/checkpoints/MRPC" --do_evaluate_sample

#2018-12-10 16:16
#python run_classifier.py --bert_model "bert-base-uncased" --data_dir "./datas/" --task_name "all" --output_dir "./checkpoints/all" --do_train --train_batch_size 128 --eval_batch_size 64 --model_suffix "bert-base-uncased_Dec11" --num_train_epochs 50 --early_training_epoch_threshold 5 --log_step 500 --resume_path "./checkpoints/all_bert-base-uncased_Dec10_Epoch7.ckpt"
#2018-12-11 12:01
CUDA_VISIBLE_DEVICES=2,3 python run_classifier.py --bert_model "bert-base-uncased" --data_dir "./datas/" --task_name "all" --output_dir "./checkpoints" --do_evaluate_sample --resume_path "./checkpoints/all_bert-base-uncased_Dec12_Epoch20.ckpt"
