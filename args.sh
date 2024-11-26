TRAINER_ARGS='--accelerator gpu --max_epochs 10 --precision 16 --num_nodes 1 --devices 1' # specify the number of gpus as '--devices'
MAIN_ARGS='--loggername tensorboard --classifier_module v6 --image_path data'
DATA_ARGS='--batch_size 16 --num_workers 2 --train_split 0.70 --val_split 0.20 --random_augment_training --augmentation_prob 1' #--limit_training_samples 2
DEFAULT_ARGS='--project_name BrainDeepLearning/Brain-DeepLearning'  
OPTIONAL_ARGS='--c_multiplier 2 --last_layer_full_MSA True --clf_head_version v1 --downstream_task tasks --num_classes 5' #--use_scheduler --gamma 0.5 --cycle 0.5'
RESUME_ARGS='' #'--load_model_path SwiFT/pretrained_models/hcp_sex_classification.ckpt'

#export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOTNlMjg1OC0yY2NmLTRjN2ItODc0NC1iMjhjNzRiZGJhMTYifQ==" # when using neptune as a logger

#export CUDA_VISIBLE_DEVICES={GPU number}

python3 main.py $TRAINER_ARGS $MAIN_ARGS $DEFAULT_ARGS $DATA_ARGS $OPTIONAL_ARGS $RESUME_ARGS \
--dataset_split_num 1 --seed 1 --learning_rate 5e-5 --model swin4d_ver7 --depth 2 2 6 2 --embed_dim 36 --max_length 12 --first_window_size 4 4 4 4 --window_size 4 4 4 4 --img_size 96 96 96 12