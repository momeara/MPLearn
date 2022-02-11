from typing import *
import glob
import functools
import copy

import numpy as np
import skimage.io
import torch
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
import torchio as tio

from .virtual_staining_dataset import VirtualStainingDataset
from .virtual_staining_queue import VirtualStainingQueue
from .uniform_area_sampler import UniformAreaSampler

class TiffReader():
    def __call__(self, field_path):
        image = skimage.io.imread(str(field_path)).astype(np.int32)
        return image, np.eye(4)


        
class VirtualStainingDataModule(pl.LightningDataModule):
    def __init__(
            self,
            dpc_images_dir: str,
            stain_images_dir: str,
            queue_length: int,
            queue_num_workers: int,
            samples_per_volume: int,
            patch_dim_x: int,
            patch_dim_y: int,            
            batch_size: int):
        super().__init__()

        self.dpc_images_dir = dpc_images_dir,
        self.stain_images_dir = stain_images_dir,
        self.dpc_field_paths = [f"{dpc_images_dir}/W0004F{F:04}T0001C1.tif" for F in range(1, 13)]
        self.stain_field_paths = [
            f"{stain_images_dir}/W0004F{F:04}T0001C1.tif"
            for F in range(1, 13)]
        assert len(self.dpc_field_paths) == len(self.stain_field_paths)

        self.queue_length = queue_length
        self.queue_num_workers = queue_num_workers
        self.samples_per_volume = samples_per_volume
        self.patch_dim_x = patch_dim_x
        self.patch_dim_y = patch_dim_y
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):        
        dpc_fields = []
        field_index = 1
        for field_path in self.dpc_field_paths:
            dpc_field = tio.Subject(
                dpc = tio.ScalarImage(path = field_path, reader = TiffReader()),
                field_index = field_index)
            dpc_fields.append(dpc_field)
            field_index += 1

        stain_fields = []
        field_index = 1
        for field_path in self.stain_field_paths:
            stain_field = tio.Subject(
                stain = tio.ScalarImage(path = field_path, reader = TiffReader()),
                field_index = field_index)
            stain_fields.append(stain_field)
            field_index += 1
            
        self.dataset = VirtualStainingDataset(
            dpc_subjects = dpc_fields,
            stain_subjects = stain_fields)
        print('Dataset size:', len(self.dataset), 'fields')

        self.dataset_train, self.dataset_val = random_split(self.dataset, [8, 4])

    def train_dataloader(self):
        sampler = UniformAreaSampler(
            patch_size = (self.patch_dim_x, self.patch_dim_y))

        patches_queue = VirtualStainingQueue(
            subject_dataset = self.dataset,
            max_length = self.queue_length,
            samples_per_volume = self.samples_per_volume,
            sampler = sampler,
            num_workers=self.queue_num_workers)
        
        return DataLoader(patches_queue, batch_size=self.batch_size)

    def val_dataloader(self):
        sampler = UniformAreaSampler(
            patch_size = (self.patch_dim_x, self.patch_dim_y))

        patches_queue = VirtualStainingQueue(
            subject_dataset = self.dataset,
            max_length = self.queue_length,
            samples_per_volume = self.samples_per_volume,
            sampler = sampler,
            num_workers=self.queue_num_workers)
        
        return DataLoader(patches_queue, batch_size=self.batch_size)

    def test_dataloader(self):
        sampler = UniformAreaSampler(
            patch_size = (self.patch_dim_x, self.patch_dim_y))

        patches_queue = VirtualStainingQueue(
            subject_dataset = self.dataset,
            max_length = self.queue_length,
            samples_per_volume = self.samples_per_volume,
            sampler = sampler,
            num_workers=self.queue_num_workers)
        
        return DataLoader(patches_queue, batch_size=self.batch_size)

    def teardown(self, stage: Optional[str] = None):
        pass
        # Used to clean-up when the run is finished
