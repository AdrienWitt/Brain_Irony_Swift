# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 21:13:55 2024

@author: adywi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np
import os
import pickle
import scipy

import torchmetrics
import torchmetrics.classification
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryROC, MulticlassAccuracy
from torchmetrics import  PearsonCorrCoef # Accuracy,
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_curve
import monai.transforms as monai_t

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import nibabel as nb


from .models.load_model import load_model
from .utils.metrics import Metrics
from .utils.parser import str2bool
from .utils.losses import NTXentLoss, global_local_temporal_contrastive
from .utils.lr_scheduler import WarmupCosineSchedule, CosineAnnealingWarmUpRestarts

from einops import rearrange

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, KBinsDiscretizer

class LitClassifier(pl.LightningModule):
    def __init__(self,data_module, **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs) # save hyperparameters except data_module (data_module cannot be pickled as a checkpoint)
       
        # you should define target_values at the Dataset classes
        target_values = data_module.train_dataset.target_values
        if self.hparams.label_scaling_method == 'standardization':
            scaler = StandardScaler()
            normalized_target_values = scaler.fit_transform(target_values)
            print(f'target_mean:{scaler.mean_[0]}, target_std:{scaler.scale_[0]}')
        elif self.hparams.label_scaling_method == 'minmax': 
            scaler = MinMaxScaler()
            normalized_target_values = scaler.fit_transform(target_values)
            print(f'target_max:{scaler.data_max_[0]},target_min:{scaler.data_min_[0]}')
        self.scaler = scaler
        print(self.hparams.model)
        self.model = load_model(self.hparams.model, self.hparams)
        
        # Heads
        if not self.hparams.pretraining:
            if self.hparams.downstream_task_type == 'classification':
                self.output_head = load_model("clf_mlp", self.hparams)
            elif self.hparams.downstream_task_type == 'regression':
                self.output_head = load_model("reg_mlp", self.hparams)
        elif self.hparams.use_contrastive:
            self.output_head = load_model("emb_mlp", self.hparams)
        else:
            raise NotImplementedError("output head should be defined")
        
        ## Loas and accuracy
        if self.hparams.downstream_task == 'literal':
            self.loss_fn = F.binary_cross_entropy_with_logits
            self.accuracy = BinaryAccuracy()
        else:
            self.loss_fn = F.cross_entropy
            self.accuracy = MulticlassAccuracy(num_classes=5)

    def forward(self, x):
        return self.output_head(self.model(x))
            
    def training_step(self, batch, batch_idx):
        fmri, subj, target, label = batch.values()
        feature = self.model(fmri)
        logits = self.output_head(feature).squeeze()
        target = target.float()
        loss = self.loss_fn(logits, target)
        self.train_acc = self.accuracy(logits, target.int())
        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('train_acc', self.train_acc, prog_bar=True, logger=True, batch_size=self.hparams.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        fmri, subj, target, label = batch.values()
        feature = self.model(fmri)
        logits = self.output_head(feature).squeeze()
        target = target.float()
        loss = self.loss_fn(logits, target)
        self.val_acc = self.accuracy(logits, target.int())
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_accuracy", self.val_acc, prog_bar=True, logger=True, batch_size=self.hparams.batch_size)
        return {"val_loss": loss, "val_acc": self.val_acc}

    def validation_epoch_end(self, outputs):
        flat_outputs = [item for sublist in outputs for item in sublist]
        avg_val_loss = torch.stack([x["val_loss"] for x in flat_outputs]).mean()
        avg_val_acc = torch.stack([x["val_acc"] for x in flat_outputs]).mean()    
        self.log('val_loss_epoch', avg_val_loss, prog_bar=True, logger=True)
        self.log('val_accuracy_epoch', avg_val_acc, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        fmri, subj, target, label = batch.values()
        feature = self.model(fmri)
        logits = self.output_head(feature).squeeze()
        target = target.float()
        loss = self.loss_fn(logits, target)
        self.test_acc = self.accuracy(logits, target.int())
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_accuracy", self.test_acc, prog_bar=True, logger=True, batch_size=self.hparams.batch_size)
        return {"test_loss": loss, "test_acc": self.test_acc}
    
    def test_epoch_end(self, outputs):
        avg_test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()    
        self.log('test_loss_epoch', avg_test_loss, prog_bar=True, logger=True)
        self.log('test_accuracy_epoch', avg_test_acc, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        if self.hparams.optimizer == "AdamW":
            optim = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == "SGD":
            optim = torch.optim.SGD(
                self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum
            )
        else:
            print("Error: Input a correct optimizer name (default: AdamW)")
        
        if self.hparams.use_scheduler:
            print()
            print("training steps: " + str(self.trainer.estimated_stepping_batches))
            print("using scheduler")
            print()
            total_iterations = self.trainer.estimated_stepping_batches # ((number of samples/batch size)/number of gpus) * num_epochs
            gamma = self.hparams.gamma
            base_lr = self.hparams.learning_rate
            warmup = int(total_iterations * 0.05) # adjust the length of warmup here.
            T_0 = int(self.hparams.cycle * total_iterations)
            T_mult = 1
            
            sche = CosineAnnealingWarmUpRestarts(optim, first_cycle_steps=T_0, cycle_mult=T_mult, max_lr=base_lr,min_lr=1e-9, warmup_steps=warmup, gamma=gamma)
            print('total iterations:',self.trainer.estimated_stepping_batches * self.hparams.max_epochs)

            scheduler = {
                "scheduler": sche,
                "name": "lr_history",
                "interval": "step",
            }

            return [optim], [scheduler]
        else:
            return optim

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("Default classifier")
        # training related
        group.add_argument("--grad_clip", action='store_true', help="whether to use gradient clipping")
        group.add_argument("--optimizer", type=str, default="AdamW", help="which optimizer to use [AdamW, SGD]")
        group.add_argument("--use_scheduler", action='store_true', help="whether to use scheduler")
        group.add_argument("--weight_decay", type=float, default=0.01, help="weight decay for optimizer")
        group.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate for optimizer")
        group.add_argument("--momentum", type=float, default=0, help="momentum for SGD")
        group.add_argument("--gamma", type=float, default=1.0, help="decay for exponential LR scheduler")
        group.add_argument("--cycle", type=float, default=0.3, help="cycle size for CosineAnnealingWarmUpRestarts")
        group.add_argument("--milestones", nargs="+", default=[100, 150], type=int, help="lr scheduler")
        group.add_argument("--adjust_thresh", action='store_true', help="whether to adjust threshold for valid/test")
        
        # pretraining-related
        group.add_argument("--use_contrastive", action='store_true', help="whether to use contrastive learning (specify --contrastive_type argument as well)")
        group.add_argument("--contrastive_type", default=0, type=int, help="combination of contrastive losses to use [1: Use the Instance contrastive loss function, 2: Use the local-local temporal contrastive loss function, 3: Use the sum of both loss functions]")
        group.add_argument("--pretraining", action='store_true', help="whether to use pretraining")
        group.add_argument("--augment_during_training", action='store_true', help="whether to augment input images during training")
        group.add_argument("--augment_only_affine", action='store_true', help="whether to only apply affine augmentation")
        group.add_argument("--augment_only_intensity", action='store_true', help="whether to only apply intensity augmentation")
        group.add_argument("--temperature", default=0.1, type=float, help="temperature for NTXentLoss")
        
        # model related
        group.add_argument("--model", type=str, default="none", help="which model to be used")
        group.add_argument("--in_chans", type=int, default=1, help="Channel size of input image")
        group.add_argument("--embed_dim", type=int, default=24, help="embedding size (recommend to use 24, 36, 48)")
        group.add_argument("--window_size", nargs="+", default=[4, 4, 4, 4], type=int, help="window size from the second layers")
        group.add_argument("--first_window_size", nargs="+", default=[2, 2, 2, 2], type=int, help="first window size")
        group.add_argument("--patch_size", nargs="+", default=[6, 6, 6, 1], type=int, help="patch size")
        group.add_argument("--depths", nargs="+", default=[2, 2, 6, 2], type=int, help="depth of layers in each stage")
        group.add_argument("--num_heads", nargs="+", default=[3, 6, 12, 24], type=int, help="The number of heads for each attention layer")
        group.add_argument("--c_multiplier", type=int, default=2, help="channel multiplier for Swin Transformer architecture")
        group.add_argument("--last_layer_full_MSA", type=str2bool, default=False, help="whether to use full-scale multi-head self-attention at the last layers")
        group.add_argument("--clf_head_version", type=str, default="v1", help="clf head version, v2 has a hidden layer")
        group.add_argument("--attn_drop_rate", type=float, default=0, help="dropout rate of attention layers")
        group.add_argument("--num_classes", type=int, default=2, help="Number of classes")

        # others
        group.add_argument("--scalability_check", action='store_true', help="whether to check scalability")
        group.add_argument("--process_code", default=None, help="Slurm code/PBS code. Use this argument if you want to save process codes to your log")
        
        return parser