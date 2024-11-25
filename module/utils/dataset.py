import os
import torch
from torch.utils.data import Dataset, IterableDataset

# import augmentations #commented out because of cv errors
import pandas as pd
from pathlib import Path
import numpy as np
import nibabel as nb
import nilearn
import random

from itertools import cycle
import glob

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, KBinsDiscretizer
import monai.transforms as monai_t
from einops import rearrange


class BaseDataset(Dataset):
    def __init__(self,  **kwargs):
        super().__init__()      
        self.register_args(**kwargs)
        self.data = self._set_data(self.subject_dict, self.max_length)
  
    def register_args(self,**kwargs):
        for name,value in kwargs.items():
            setattr(self,name,value)
        self.kwargs = kwargs
        
    def load_random_sequence(self, data, file, subject, target):
        if self.use_contrastive:    
            same_subject_target_samples = [t for t in data if t[1] == subject and t[6] == target and t[2] != file]
            if same_subject_target_samples:
                rand_sample = random.choice(same_subject_target_samples)
                return rand_sample[3]
    
    def pad_back_forward(self, length, desired_length):
        pad_back = (desired_length-length)//2
        pad_forward = desired_length-length-pad_back
        return (pad_back, pad_forward)
    
    def pad_to_max(self, img):
        y = img
        y_x, y_y, y_z, y_t = y.size(1), y.size(2), y.size(3), y.size(4)
        x_back, x_for = self.pad_back_forward(y_x, self.img_size[0])
        y_back, y_for = self.pad_back_forward(y_y, self.img_size[1])
        z_back, z_for = self.pad_back_forward(y_z, self.img_size[2])
        t_back, t_for = self.pad_back_forward(y_t, self.max_length)
        background_value = y.flatten()[0]
        y = y.permute(0,4,1,2,3)
        y = torch.nn.functional.pad(y, (z_back, z_for, y_back, y_for, x_back, x_for, t_back, t_for), value=background_value)
        y = y.permute(0,2,3,4,1)
        return y
    
    def augment(self, img):

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
        if self.train_augment_only_intensity:
            comp = monai_t.Compose([rand_noise, rand_smooth])
        else:
            comp = monai_t.Compose([rand_affine, rand_noise, rand_smooth]) 

            aug_seed = torch.randint(0, 10000000, (1,)).item()
            # set augmentation seed to be the same for all time steps
            for t in range(T):
                if self.train_augment_only_affine:
                    rand_affine.set_random_state(seed=aug_seed)
                    img[t, :, :, :, :] = rand_affine(img[t, :, :, :, :])
                else:
                    comp.set_random_state(seed=aug_seed)
                    img[t, :, :, :, :] = comp(img[t, :, :, :, :])

        img = rearrange(img, 't c h w d -> c h w d t')

        return img
    
    def _set_data(self, subject_dict, max_length):
        data = []
        for i, subject in enumerate(subject_dict):
            file_dicts = subject_dict[subject]
            for file_dict in file_dicts:
                subject = file_dict["subject"]
                file = file_dict["file"]
                sequence_length = file_dict["sequence_length"]
                target = file_dict["target"]
                label = file_dict["label"]
                data_tuple = (i, subject, file, sequence_length, max_length, target, label)
                data.append(data_tuple)
                if self.use_augmentation and torch.rand(1).item() < self.augmentation_prob:
                   data_augmented_tuple = (i, subject, file, sequence_length, max_length, target, label, True) ## add flag for augmented
                   data.append(data_augmented_tuple)    
        
        if self.train : 
            self.target_values = np.array([tup[5] for tup in data]).reshape(-1, 1)
        
        return data

    def __getitem__(self, index):
        sample = self.data[index]
        if len(sample) == 8:  # Augmented sample with an extra flag
            _, subject, file, sequence_length, max_length, target, label, is_augmented = sample
        else:
            _, subject, file, sequence_length, max_length, target, label = sample
            is_augmented = False
        
        y = torch.load(file).unsqueeze(0)

        if is_augmented:
            y = self.augment(y)
        
        y = self.pad_to_max(y)
        
        if self.use_contrastive:
            rand_img = self.load_random_sequence(self.data, file, subject, target)
            rand_y = self.pad_to_max(rand_img)
            return {
            "fmri_sequence": (y, rand_y),
            "subject_name": subject,
            "target": target,
            "label": label}
        
        return {
            "fmri_sequence": y,
            "subject_name": subject,
            "target": target,
            "label": label}
            
    def __len__(self):
        return  len(self.data)
    
    

