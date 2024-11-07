import os
import glob
import numpy as np
import nibabel as nib
from nilearn.image import concat_imgs
import pandas as pd
from nilearn.image import crop_img, mean_img, iter_img, math_img
import matplotlib.pyplot as plt
from nilearn.masking import compute_epi_mask
import torch

root_folder = r'D:\Preproc_Analyses\data_done'
files_type = ['wrMF']
output_dir = r'C:\Users\wittmann\OneDrive - unige.ch\Documents\Sarcasm_experiment\Brain_DeepLearning\Brain_Irony_Swift\data'

def select_files(root_folder, files_type):
    participant_folders = glob.glob(os.path.join(root_folder, 'p*'))
    participant_files = {}  
    for participant_folder in participant_folders:
        participant = participant_folder[-3:]
        run_folders = glob.glob(os.path.join(participant_folder, 'RUN*'))
        run_files = {}
        for run_folder in run_folders:
                run = run_folder[-4:]
                nii_files = glob.glob(os.path.join(run_folder, f'{files_type}*.nii'))                
                run_files[run]  = nii_files
                participant_files[participant] = run_files
    return participant_files
           
def save_concatenated(file_type, root_folder, output_dir):
    participant_files = select_files()
    for participant, runs in participant_files.items():
        for run in runs.values():
            concatenated_img = concat_imgs(run['files'])
            output_filename = f'{file_type}_{participant}_{run}'
            output_dir = os.path.join(root_folder, output_dir)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_path = os.path.join(output_dir, output_filename)
            nib.save(concatenated_img, output_path)
            
def load_dataframe(participant_path):
    file_list = [f for f in os.listdir(participant_path) if f.startswith('Resultfile')]
    dfs = {}
    for file_name in file_list:
        full_path = os.path.join(participant_path, file_name)
        df = pd.read_csv(full_path, sep='\t') 
        key = file_name[-8:-4]
        dfs[key] = df
    return dfs

def crop_skull_background(fmri):
    mask = compute_epi_mask(fmri)
    masked_list = []
    for img in iter_img(fmri):
            masked = math_img('img*mask', img = img, mask=mask)
            masked_list.append(masked)
    masked_concat = concat_imgs(masked_list)
    croped_img = crop_img(masked_concat)
    return croped_img

def mean_z_norm_to_tensor(fmri):
    fmri = torch.Tensor(fmri.get_fdata())
    global_mean = fmri.mean()
    print("mean:", global_mean)
    global_std = fmri.std()
    print("std:", global_std)
    fmri_temp = (fmri - global_mean) / global_std
    return fmri_temp

for file_type in files_type:
    participant_files = select_files(root_folder, file_type)
    
    for participant, runs in participant_files.items():
        dfs = load_dataframe(os.path.join(root_folder, participant))
        
        for run_number, run, df in zip(runs.keys(), runs.values(), dfs.values()): 
            concatenated_img = concat_imgs(run)
            croped_img = crop_skull_background(concatenated_img)
            tensor_img = mean_z_norm_to_tensor(croped_img)
            df = df.rename(columns=lambda x: x.strip())
            for i, row in df.iterrows():
               condition_name = row['Condition_name']
               situation = row['Situation']
               task = row['task']
               start = row['Real_Time_Onset_Statement']
               if pd.isna(row['Real_Time_End_Evaluation']):
                    end = row['Real_Time_Onset_Evaluation'] + 5
               else:
                    end = row['Real_Time_End_Evaluation']
               
               print(start)
               print(end)
               start_scan = round(start / 0.65)
               end_scan = round(end / 0.65)
               scans = tensor_img[..., start_scan:end_scan]
               scans = scans.type(torch.float16)
               background_value = scans.flatten()[0]
               filename = f'{participant}_{file_type}_{run_number}_{task}_{situation}_{condition_name}'
               subj_dir = os.path.join(output_dir, participant)
               
               if not os.path.exists(subj_dir):
                   os.makedirs(subj_dir)
               torch.save(scans.clone(), os.path.join(subj_dir,filename +".pt"))
               print(os.path.join(subj_dir,filename +".pt"))
      