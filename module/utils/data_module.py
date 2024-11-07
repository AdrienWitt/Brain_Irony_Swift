# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 17:09:37 2024

@author: adywi
"""

import os
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from .dataset import BaseDataset
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from .parser import str2bool

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

import torch

class fMRIDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        if self.hparams.use_custom_max_length:
            self.max_length = self.hparams.max_length
        else:
            self.max_length = 0
        
        split_dir_path = f'splits/{self.hparams.downstream_task}'
        os.makedirs(split_dir_path, exist_ok=True)
        self.split_file_path = os.path.join(split_dir_path, f"split_fixed_{self.hparams.dataset_split_num}.txt")
        
        self.setup()

        #pl.seed_everything(seed=self.hparams.data_seed)

    def get_dataset(self):
            return BaseDataset

    def convert_subject_list_to_idx_list(self, train_names, val_names, test_names, subj_list):
        #subj_idx = np.array([str(x[0]) for x in subj_list])
        subj_idx = np.array([str(x[1]) for x in subj_list])
        S = np.unique([x[1] for x in subj_list])
        # print(S)
        print('unique subjects:',len(S))  
        train_idx = np.where(np.in1d(subj_idx, train_names))[0].tolist()
        val_idx = np.where(np.in1d(subj_idx, val_names))[0].tolist()
        test_idx = np.where(np.in1d(subj_idx, test_names))[0].tolist()
        return train_idx, val_idx, test_idx
    
    def save_split(self, sets_dict):
        with open(self.split_file_path, "w+") as f:
            for name, subj_list in sets_dict.items():
                f.write(name + "\n")
                for subj_name in subj_list:
                    f.write(str(subj_name) + "\n")
                    
    def determine_split_randomly(self, S):
        S = list(S.keys())
        S_train = int(len(S) * self.hparams.train_split)
        S_val = int(len(S) * self.hparams.val_split)
        S_train = np.random.choice(S, S_train, replace=False)
        remaining = np.setdiff1d(S, S_train) # np.setdiff1d(np.arange(S), S_train)
        S_val = np.random.choice(remaining, S_val, replace=False)
        S_test = np.setdiff1d(S, np.concatenate([S_train, S_val])) # np.setdiff1d(np.arange(S), np.concatenate([S_train, S_val]))
        # train_idx, val_idx, test_idx = self.convert_subject_list_to_idx_list(S_train, S_val, S_test, self.subject_list)
        self.save_split({"train_subjects": S_train, "val_subjects": S_val, "test_subjects": S_test})
        return S_train, S_val, S_test
    
    def load_split(self):
        subject_order = open(self.split_file_path, "r").readlines()
        subject_order = [x[:-1] for x in subject_order]
        train_index = np.argmax(["train" in line for line in subject_order])
        val_index = np.argmax(["val" in line for line in subject_order])
        test_index = np.argmax(["test" in line for line in subject_order])
        train_names = subject_order[train_index + 1 : val_index]
        val_names = subject_order[val_index + 1 : test_index]
        test_names = subject_order[test_index + 1 :]
        return train_names, val_names, test_names

    def prepare_data(self):
        # This function is only called at global rank==0
        return
    
    def make_subject_dict(self):
        img_root = self.hparams.image_path
        final_dict = dict()
        subject_list = os.listdir(img_root)
        
        for subject in subject_list:
            subject_path = os.path.join(img_root, subject)
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
                
                if not self.hparams.use_custom_max_length:
                    self.max_length = max(self.max_length, sequence_length)
  
                if self.hparams.downstream_task == 'literal':
                    target = 1 if condition in ["CN_SPneg", "CP_SNpos"] else 0
                    label = condition
                elif self.hparams.downstream_task == 'tasks':
                    target = (0 if task == 'prosody' else
                                       1 if task == 'semantic' else
                                       2 if task == 'irony' else
                                       3 if task == 'sarcasm' else
                                       5)
                    label = task      
                file_dict = {
                        'subject' : subject,
                        'file': file_path,
                        'file_type': file_type,
                        "sequence_length" : sequence_length,
                        'target': target,
                        'label': label}
                file_dicts.append(file_dict)
                final_dict[subject]=file_dicts
        return final_dict
            
    def setup(self, stage=None):
        # this function will be called at each devices
        Dataset = self.get_dataset()
        subject_dict = self.make_subject_dict()
        
        params = {
                "root": self.hparams.image_path,
                "max_length": self.max_length,
                "img_size": self.hparams.img_size,
                "downstream_task": self.hparams.downstream_task,
                "use_contrastive" : self.hparams.use_contrastive,
                "contrastive_type" : self.hparams.contrastive_type,
                "dtype":'float16'}
        
        if os.path.exists(self.split_file_path):
            train_names, val_names, test_names = self.load_split()
        else:
            train_names, val_names, test_names = self.determine_split_randomly(subject_dict)
                
        if self.hparams.limit_training_samples:
            train_names = np.random.choice(train_names, size=self.hparams.limit_training_samples, replace=False, p=None)
        
        train_dict = {key: subject_dict[key] for key in train_names if key in subject_dict}
        val_dict = {key: subject_dict[key] for key in val_names if key in subject_dict}
        test_dict = {key: subject_dict[key] for key in test_names if key in subject_dict}
        
        self.train_dataset = Dataset(**params,subject_dict=train_dict,use_augmentations=False, train=True)
        # load train mean/std of target labels to val/test dataloader
        self.val_dataset = Dataset(**params,subject_dict=val_dict,use_augmentations=False,train=False) 
        self.test_dataset = Dataset(**params,subject_dict=test_dict,use_augmentations=False,train=False) 
        
        print("number of train_subj:", len(train_dict))
        print("number of val_subj:", len(val_dict))
        print("number of test_subj:", len(test_dict))
        print("length of train_idx:", len(self.train_dataset.data))
        print("length of val_idx:", len(self.val_dataset.data))  
        print("length of test_idx:", len(self.test_dataset.data))
        
        # DistributedSampler is internally called in pl.Trainer
        def get_params(train):
            return {
                "batch_size": self.hparams.batch_size if train else self.hparams.eval_batch_size,
                "num_workers": self.hparams.num_workers,
                "drop_last": True,
                "pin_memory": False,
                "persistent_workers": True if (train and (self.hparams.strategy == 'ddp')) else False,
                "shuffle": train
            }
        self.train_loader = DataLoader(self.train_dataset, **get_params(train=True))
        self.val_loader = DataLoader(self.val_dataset, **get_params(train=False))
        self.test_loader = DataLoader(self.test_dataset, **get_params(train=False))
        

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        # return self.val_loader
        # currently returns validation and test set to track them during training
        return [self.val_loader, self.test_loader]

    def test_dataloader(self):
        return self.test_loader

    def predict_dataloader(self):
        return self.test_dataloader()

    @classmethod
    def add_data_specific_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=True, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("DataModule arguments")
        group.add_argument("--dataset_split_num", type=int, default=1) # dataset split, choose from 1, 2, or 3
        group.add_argument("--image_path", default=None, help="path to image datasets preprocessed for SwiFT")
        group.add_argument("--train_split", default=0.7, type=float)
        group.add_argument("--val_split", default=0.15, type=float)
        group.add_argument("--batch_size", type=int, default=4)
        group.add_argument("--eval_batch_size", type=int, default=16)
        group.add_argument("--img_size", nargs="+", default=[96, 96, 96, 12], type=int, help="image size (adjust the fourth dimension according to your --max_length argument)")
        group.add_argument("--max_length", type=int, default=12)
        group.add_argument("--use_custom_max_length", type=str2bool, nargs='?', const=True, default=False, help="Use custom max_length from data (default: False)")
        group.add_argument("--num_workers", type=int, default=8)
        group.add_argument("--limit_training_samples", type=int, default=None, help="use if you want to limit training samples")
        group.add_argument("--label_scaling_method", default="standardization", choices=["minmax","standardization"], help="label normalization strategy for a regression task (mean and std are automatically calculated using train set)")

        return parser
