# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 15:02:07 2023

@author: adywi
"""

import os
import torch
from torch.utils.data import Dataset, IterableDataset
import pandas as pd
from pathlib import Path
import numpy as np
import nibabel as nb
import nilearn
import random


image_path = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Brain_DeepLearning\SwiFT\SwiFT_tutorial_official\MNI_to_TRs"
dataset_name = "S1200"
downstream_task = 'sex'
img_root = os.path.join(image_path, 'img')
start_frame = 1
sample_duration = 20
stride_within_seq = 1
stride_between_seq = 1
stride = max(round(stride_between_seq * sample_duration),1)
with_voxel_norm = True


img_root = os.path.join(image_path, 'img')
final_dict = dict()
if dataset_name == "S1200":
    subject_list = os.listdir(img_root)
    meta_data = pd.read_csv(os.path.join(image_path, "metadata", "HCP_1200_gender.csv"))
    if downstream_task == 'sex': task_name = 'Gender'
    else: raise NotImplementedError()

    if downstream_task == 'sex':
        meta_task = meta_data[['Subject',task_name]].dropna()

    for subject in subject_list:
        if subject in meta_task['Subject'].values:
            if downstream_task == 'sex':
                target = meta_task[meta_task["Subject"]==subject][task_name].values[0]
                target = 1 if target == "M" else 0
                sex = target
            final_dict[subject]=[sex,target]

subject_dict = final_dict

def save_split(sets_dict):
    with open(split_file_path, "w+") as f:
        for name, subj_list in sets_dict.items():
            f.write(name + "\n")
            for subj_name in subj_list:
                f.write(str(subj_name) + "\n")

def determine_split_randomly(S):
    S = list(S.keys())
    S_train = int(len(S) * train_split)
    S_val = int(len(S) * val_split)
    S_train = np.random.choice(S, S_train, replace=False)
    remaining = np.setdiff1d(S, S_train) # np.setdiff1d(np.arange(S), S_train)
    S_val = np.random.choice(remaining, S_val, replace=False)
    S_test = np.setdiff1d(S, np.concatenate([S_train, S_val])) # np.setdiff1d(np.arange(S), np.concatenate([S_train, S_val]))
    # train_idx, val_idx, test_idx = convert_subject_list_to_idx_list(S_train, S_val, S_test, subject_list)
    save_split({"train_subjects": S_train, "val_subjects": S_val, "test_subjects": S_test})
    return S_train, S_val, S_test

def load_split():
    subject_order = open(split_file_path, "r").readlines()
    subject_order = [x[:-1] for x in subject_order]
    train_index = np.argmax(["train" in line for line in subject_order])
    val_index = np.argmax(["val" in line for line in subject_order])
    test_index = np.argmax(["test" in line for line in subject_order])
    train_names = subject_order[train_index + 1 : val_index]
    val_names = subject_order[val_index + 1 : test_index]
    test_names = subject_order[test_index + 1 :]
    return train_names, val_names, test_names


split_dir_path = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Brain_DeepLearning\SwiFT\SwiFT_tutorial_official\SwiFT\spit_test"
os.makedirs(split_dir_path, exist_ok=True)
split_file_path = os.path.join(split_dir_path, f"split_fixed_{1}.txt")

train_names, val_names, test_names = determine_split_randomly(subject_dict)
train_names, val_names, test_names = load_split()


train_dict = {key: subject_dict[key] for key in train_names if key in subject_dict}
val_dict = {key: subject_dict[key] for key in val_names if key in subject_dict}
test_dict = {key: subject_dict[key] for key in test_names if key in subject_dict}
os.chdir(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Brain_DeepLearning\SwiFT\SwiFT_tutorial_official\SwiFT\project\module\utils")

from data_preprocess_and_load.datasets import S1200


params = {
                "root": image_path,
                "sequence_length": 20,
                "contrastive":False,
                "stride_between_seq": stride_between_seq,
                "stride_within_seq": stride_within_seq,
                "with_voxel_norm": with_voxel_norm,
                "downstream_task": downstream_task,
                "shuffle_time_sequence": False,
                "dtype":'float16'}




train_dataset = Dataset(**params,subject_dict=train_dict,use_augmentations=False, train=True)
# load train mean/std of target labels to val/test dataloader
val_dataset = Dataset(**params,subject_dict=val_dict,use_augmentations=False,train=False) 
test_dataset = Dataset(**params,subject_dict=test_dict,use_augmentations=False,train=False) 


################################################################################################################################################
subject_path = os.path.join(img_root, subject_list[0])
train = True

y = []       
load_fnames = [f'frame_{frame}.pt' for frame in range(start_frame, start_frame+sample_duration,stride_within_seq)]
for fname in load_fnames:
    img_path = os.path.join(subject_path, fname)
    y_i = torch.load(img_path).unsqueeze(0)
    y.append(y_i)
y = torch.cat(y, dim=4)

root = img_root
subject_dict = final_dict

data = []
for i, subject in enumerate(os.listdir(root)):
    sex,target = subject_dict[subject]
    subject_path = os.path.join(img_root, subject)
    num_frames = len(os.listdir(subject_path)) # voxel mean & std
    session_duration = num_frames - sample_duration + 1
    for start_frame in range(0, session_duration, stride):
        data_tuple = (i, subject, subject_path, start_frame, stride, num_frames, target, sex)
        data.append(data_tuple)
        data_old = data

    if train: 
        target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)


_, subject, subject_path, start_frame, sequence_length, num_frames, target, sex = data[1]

background_value = y.flatten()[0]
y = y.permute(0,4,1,2,3)
y = torch.nn.functional.pad(y, (9, 10, 3, 3, 11, 12), value=background_value)
y = y.permute(0,2,3,4,1)
with_voxel_norm = True

num_frames = len(os.listdir(subject_path))
y = []
load_fnames = [f'frame_{frame}.pt' for frame in range(start_frame, start_frame+sample_duration,stride_within_seq)]
if  with_voxel_norm:
    load_fnames += ['voxel_mean.pt', 'voxel_std.pt']

for fname in load_fnames:
    img_path = os.path.join(subject_path, fname)
    y_loaded = torch.load(img_path).unsqueeze(0)
    y.append(y_loaded)
y = torch.cat(y, dim=4)


# random_y = []
# full_range = np.arange(0, num_frames-sample_duration+1)
# # exclude overlapping sub-sequences within a subject
# exclude_range = np.arange(start_frame-sample_duration, start_frame+sample_duration)
# available_choices = np.setdiff1d(full_range, exclude_range)
# random_start_frame = np.random.choice(available_choices, size=1, replace=False)[0]
# load_fnames = [f'frame_{frame}.pt' for frame in range(random_start_frame, random_start_frame+sample_duration, stride_within_seq)]
# if with_voxel_norm:
#     load_fnames += ['voxel_mean.pt', 'voxel_std.pt']
# for fname in load_fnames:
#     img_path = os.path.join(subject_path, fname)
#     y_loaded = torch.load(img_path).unsqueeze(0)
#     random_y.append(y_loaded)
# random_y = torch.cat(random_y, dim=4)



#############################################################################################################################
root = "data"
downstream_task = 'literal'
train = True

def make_subject_dict():
    img_root = root
    max_length = 0
    final_dict = dict()
    subject_list = os.listdir(root)
    
    for subject in subject_list:
        subject_path = os.path.join(root, subject)
        file_list = os.listdir(subject_path)
        file_dicts = []

        for file in file_list:
            file_path = os.path.join(subject_path, file)
            basename = file.split('.')[0]
            parts = basename.split('_')
            file_type = parts[1]
            task = parts[3]      
            condition = f'{parts[5]}_{parts[6]}'
            sequence_length = torch.load(file_path).shape[3]
            
            if sequence_length > max_length:
                max_length = sequence_length
            if downstream_task == 'literal':
                target = 1 if condition in ["CN_SPneg", "CP_SN_pos"] else 0
            elif downstream_task == 'tasks':
                target = (0 if task == 'prosody' else
                                   1 if task == 'semantic' else
                                   2 if task == 'irony' else
                                   3 if task == 'sarcasm' else
                                   5)
            
            file_dict = {
                    'subject' : subject,
                    'file': file_path,
                    'file_type': file_type,
                    "sequence_length" : sequence_length,
                    'max_length': max_length,
                    'target': [target, target]}
            file_dicts.append(file_dict)
            final_dict[subject]=file_dicts
    
    ## Change max_length for each subject so it correspond to the actual max_length
    for subject, file_dicts in final_dict.items():
        for file_dict in file_dicts:
            file_dict['max_length'] = max_length
    return final_dict
               
subject_dict = make_subject_dict()     
use_augmentation=True

def _set_data(subject_dict):
    data = []
    for i, subject in enumerate(subject_dict):
        file_dicts = subject_dict[subject]
        for file_dict in file_dicts:
            file = file_dict["file"]
            file_type = file_dict["file_type"]
            sequence_length = file_dict["sequence_length"]
            max_length = file_dict["max_length"]
            label, target = file_dict["target"]
            
            if not use_augmentation and file_type == "swrMF":
                continue
            
            data_tuple = (i, file_dict["subject"], file, file_dict["file_type"], file_dict["sequence_length"], max_length, target, label)
            data.append(data_tuple)
                  
            if train : 
                target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)
            return data



data = _set_data(subject_dict)

from module.utils.dataset import BaseDataset

dataset = BaseDataset(**params, subject_dict = subject_dict)

train_split = 0.7
test_split = 0.15
val_split = 0.15

split_dir_path = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Brain_DeepLearning\split"

os.makedirs(split_dir_path, exist_ok=True)
split_file_path = os.path.join(split_dir_path, f"split_fixed_{1}.txt")


from torch.utils.data import DataLoader

def setup(stage=None):
    # this function will be called at each devices
    Dataset = dataset
    params = {
            "root": root,
            "max_length": 12,
            "downstream_task": 'literal',
            "dtype":'float16'}
    
    subject_dict = make_subject_dict()
    if os.path.exists(split_file_path):
        train_names, val_names, test_names = load_split()
    else:
        train_names, val_names, test_names = determine_split_randomly(subject_dict)
            

    train_dict = {key: subject_dict[key] for key in train_names if key in subject_dict}
    val_dict = {key: subject_dict[key] for key in val_names if key in subject_dict}
    test_dict = {key: subject_dict[key] for key in test_names if key in subject_dict}
    
    train_dataset = Dataset(**params,subject_dict=train_dict,use_augmentations=False, train=True)
    # load train mean/std of target labels to val/test dataloader
    val_dataset = Dataset(**params,subject_dict=val_dict,use_augmentations=False,train=False) 
    test_dataset = Dataset(**params,subject_dict=test_dict,use_augmentations=False,train=False) 
    
    print("number of train_subj:", len(train_dict))
    print("number of val_subj:", len(val_dict))
    print("number of test_subj:", len(test_dict))
    print("length of train_idx:", len(train_dataset.data))
    print("length of val_idx:", len(val_dataset.data))  
    print("length of test_idx:", len(test_dataset.data))
    
    # DistributedSampler is internally called in pl.Trainer
    def get_params(train):
        return {
            "batch_size": 12 if train else 4,
            "num_workers": 8,
            "drop_last": True,
            "pin_memory": False,
            "persistent_workers": True if (train and (strategy == 'ddp')) else False,
            "shuffle": train
        }
    train_loader = DataLoader(train_dataset, **get_params(train=True))
    val_loader = DataLoader(val_dataset, **get_params(train=False))
    test_loader = DataLoader(test_dataset, **get_params(train=False))


#########################################################################################################################
import os
os.chdir(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Brain_DeepLearning\Brain_Irony_Swift")
#root = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Brain_DeepLearning\Brain_Irony_Swift\data"
root = "data"

from module.utils.data_module import fMRIDataModule
from module.utils.dataset import BaseDataset
import torch


import random
random.seed(1234)
import torch
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)

params = {
        'image_path' : root,
        'use_contrastive' : False,
        'train': True,
        'use_augmentation': False,
        "max_length" : 0,
        'downstream_task': 'literal',
        'image_path' : root,
        'dataset_split_num': 1,
        'use_custom_max_length': False,
        "limit_training_samples" : False,
        'batch_size': 1,
        'eval_batch_size' : 16,
        'train_split': 0.7,
        'val_split': 0.2,
        "test_split" : 0.15,
        'img_size': [96, 96, 96, 12],
        'num_workers' : 8,
        'strategy' : 'ddp',
        'use_custom_max_length': False,
        'contrastive_type' : 1,
        'random_augment_training' : False,
        "augmentation_prob" : 0.5,
        "train_augment_only_affine" : False,
        "train_augment_only_intensity" : False,     
        }

data = fMRIDataModule(**params)

def make_subject_dict():
    use_custom_max_length = False
    max_length = 12
    img_root = root
    final_dict = dict()
    subject_list = os.listdir(img_root)
    downstream_task = 'literal'
    
    for subject in subject_list:
        subject_path = os.path.join(img_root, subject)
        file_list = os.listdir(subject_path)
        file_dicts = []

        for file in file_list:
            file_path = os.path.join(subject_path, file)
            basename = file.split('.')[0]
            parts = basename.split('_')
            task = parts[3]      
            condition = f'{parts[5]}_{parts[6]}'
            sequence_length = torch.load(file_path).shape[3]
            
            if not use_custom_max_length:
                max_length = max(max_length, sequence_length)

            if downstream_task == 'literal':
                target = 1 if condition in ["CN_SPneg", "CP_SNpos"] else 0
                label = condition
            elif downstream_task == 'tasks':
                target = (0 if task == 'prosody' else
                                   1 if task == 'semantic' else
                                   2 if task == 'irony' else
                                   3 if task == 'sarcasm' else
                                   5)
                label = task      
            file_dict = {
                    'subject' : subject,
                    'file': file_path,
                    "sequence_length" : sequence_length,
                    'target': target,
                    'label': label}
            file_dicts.append(file_dict)
            final_dict[subject]=file_dicts
    return final_dict


sub_dict = make_subject_dict()


dataset = BaseDataset(**params, subject_dict = sub_dict)
























# Iterate through the DataLoader until reaching the target batch
for batch_idx, batch in enumerate(data_loader):
        fmri, subject, target, label, file = batch.values()
        print(file)




target_index = processed_files.index("data_deeplearning\p12\p12_wrMF_RUN2_tom_9_CP_SPpos.pt")

from einops import rearrange
import monai.transforms as monai_t

img = torch.load("data\p12\p12_wrMF_RUN2_tom_9_CP_SPpos.pt")
train_augment_only_affine = False
train_augment_only_intensity = False
     

def augment(img):


    C, H, W, D, T = img.shape

    device = img.device
    img = rearrange(img, 'c h w d t -> t c h w d')

    rand_affine = monai_t.RandAffine(
        prob=1.0,
        # 0.175 rad = 10 degrees
        rotate_range=(0.175, 0.175, 0.175),
        scale_range = (0.1, 0.1, 0.1),
        mode = "bilinear",
        padding_mode = "border",
        device = device
    )
    rand_noise = monai_t.RandGaussianNoise(prob=0.3, std=0.1)
    rand_smooth = monai_t.RandGaussianSmooth(sigma_x=(0.0, 0.5), sigma_y=(0.0, 0.5), sigma_z=(0.0, 0.5), prob=0.1)
    if train_augment_only_intensity:
        comp = monai_t.Compose([rand_noise, rand_smooth])
    else:
        comp = monai_t.Compose([rand_affine, rand_noise, rand_smooth]) 

        aug_seed = torch.randint(0, 10000000, (1,)).item()
        # set augmentation seed to be the same for all time steps
        for t in range(T):
            if train_augment_only_affine:
                rand_affine.set_random_state(seed=aug_seed)
                img[t, :, :, :, :] = rand_affine(img[t, :, :, :, :])
            else:
                comp.set_random_state(seed=aug_seed)
                img[t, :, :, :, :] = comp(img[t, :, :, :, :])

    img = rearrange(img, 't c h w d -> c h w d t')

    return img


y = torch.load("data\p01\p01_wrMF_RUN1_sarcasm_10_CN_SNneg.pt").unsqueeze(0)
y_augment = augment(y)

y = torch.load("data_deeplearning\p01\p01_wrMF_RUN1_sarcasm_10_CN_SNneg.pt").unsqueeze(0)

for idx, sample in enumerate(dataset):
    # Get the first item from the dictionary
    first_key = list(sample.keys())[0]
    tensor = sample[first_key]  # Access the tensor associated with the first key

    # Check if there are any NaN values in the tensor
    if torch.isnan(tensor).any():
        print(f"NaN found in sample {idx}, key '{first_key}'")
    else:
        print(f"No NaN in sample {idx}, key '{first_key}'")




###################################################

os.chdir(r'C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Brain_DeepLearning\SwiFT\SwiFT_tutorial_official\SwiFT\project')

import torch
from module.utils.data_module import fMRIDataModule



pretrained_ckpt_path=r'C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Brain_DeepLearning\SwiFT\SwiFT_tutorial_official\SwiFT\pretrained_models\hcp_sex_classification_new.ckpt'
ckpt = torch.load(pretrained_ckpt_path, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')

ckpt['hyper_parameters']['image_path'] = r'C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Brain_DeepLearning\SwiFT\SwiFT_tutorial_official\MNI_to_TRs/' #HCP
#ckpt['hyper_parameters']['default_root_dir'] = ''
ckpt['hyper_parameters']['shuffle_time_sequence'] = False
ckpt['hyper_parameters']['batch_size'] = 4
ckpt['hyper_parameters']['time_as_channel'] = False
ckpt['hyper_parameters']['eval_batch_size'] = 1

args = ckpt['hyper_parameters']

data_module = fMRIDataModule(**args)
dataset = data_module.test_dataset
sample = dataset.__getitem__(1)

train_loader = data_module.train_loader
batch = next(iter(train_loader)) 

############################################################################################################
import os
os.chdir(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Brain_DeepLearning\Brain_Irony_Swift")

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from collections import OrderedDict
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger

# from module import LitClassifier
import neptune.new as neptune
from module.utils.data_module import fMRIDataModule
from module.utils.parser import str2bool
from module.pl_classifier import LitClassifier


# Set classifier
Classifier = LitClassifier

# Set dataset
Dataset = fMRIDataModule


parser = ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--seed", default=1234, type=int, help="random seeds. recommend aligning this argument with data split number to control randomness")
parser.add_argument("--downstream_task", type=str, default="literal", help="downstream task")
parser.add_argument("--classifier_module", default="default", type=str, help="A name of lightning classifier module (outdated argument)")
parser.add_argument("--loggername", default="default", type=str, help="A name of logger")
parser.add_argument("--project_name", default="default", type=str, help="A name of project (Neptune)")
parser.add_argument("--resume_ckpt_path", type=str, help="A path to previous checkpoint. Use when you want to continue the training from the previous checkpoints")
parser.add_argument("--load_model_path", type=str, help="A path to the pre-trained model weight file (.pth)")
parser.add_argument("--test_only", action='store_true', help="specify when you want to test the checkpoints (model weights)")
parser.add_argument("--test_ckpt_path", type=str, help="A path to the previous checkpoint that intends to evaluate (--test_only should be True)")
parser.add_argument("--freeze_feature_extractor", action='store_true', help="Whether to freeze the feature extractor (for evaluating the pre-trained weight)")
temp_args, _ = parser.parse_known_args()

# add two additional arguments
parser = Classifier.add_model_specific_args(parser)
parser = Dataset.add_data_specific_args(parser)

_, _ = parser.parse_known_args()  # This command blocks the help message of Trainer class.
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

#override parameters
max_epochs = args.max_epochs
num_nodes = args.num_nodes
devices = args.devices
project_name = args.project_name
image_path = args.image_path

if temp_args.resume_ckpt_path is not None:
    # resume previous experiment
    from module.utils.neptune_utils import get_prev_args
    args = get_prev_args(args.resume_ckpt_path, args)
    exp_id = args.id
    # override max_epochs if you hope to prolong the training
    args.project_name = project_name
    args.max_epochs = max_epochs
    args.num_nodes = num_nodes
    args.devices = devices
    args.image_path = image_path       
else:
    exp_id = None

setattr(args, "default_root_dir", f"output/{args.project_name}")


    



