TRAINER_ARGS='--accelerator gpu --max_epochs 10 --precision 16 --num_nodes 1 --devices 1' # specify the number of gpus as '--devices'
MAIN_ARGS='--loggername tensorboard --classifier_module v6 --image_path data'
DATA_ARGS='--batch_size 16 --num_workers 2 --train_split 0.3333333333333333 --val_split 0.3333333333333333'
DEFAULT_ARGS='' #--project_name '
OPTIONAL_ARGS='--c_multiplier 2 --last_layer_full_MSA True --clf_head_version v1 --downstream_task literal \ --pretraining --use_contrastive --contrastive_type 1'
RESUME_ARGS=''

#export NEPTUNE_API_TOKEN="{neptune API token}" # when using neptune as a logger

#export CUDA_VISIBLE_DEVICES={GPU number}

python3 main.py $TRAINER_ARGS $MAIN_ARGS $DEFAULT_ARGS $DATA_ARGS $OPTIONAL_ARGS $RESUME_ARGS \
--dataset_split_num 1 --seed 1 --learning_rate 5e-5 --model swin4d_ver7 --depth 2 2 6 2 --embed_dim 36 --max_length 12 --first_window_size 4 4 4 4 --window_size 4 4 4 4 --img_size 96 96 96 12