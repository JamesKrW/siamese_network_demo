import math
from abc import ABC
from typing import List, Union

from torch.utils.data import Dataset

from numpy import ndarray
import argparse, random, copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T
from torch.optim.lr_scheduler import StepLR
import sys

class BaseDataset(ABC, Dataset):
    FEATURE_COLUMNS: List[str]
    targets: Union[List[int], ndarray]

    @property
    def features(self) -> List[str]:
        return self.FEATURE_COLUMNS

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def n_classes(self) -> Union[float, int]:
        return (
            math.inf
        )  # infinity means regression, it is the default

import logging
import math
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


from split import split_evenly_by_classes

log = logging.getLogger(__name__)


class ClassificationDataset(BaseDataset):
    TRAIN_RATIO = 0.8
    DEFAULT_SEED = 999

    FEATURE_COLUMNS = [
        "io_time",
    ]

    def __init__(
        self,
        filename: Union[str, Path],
        train_ratio: float = TRAIN_RATIO,
        train: bool = False,
        seed: Optional[int] = DEFAULT_SEED,
    ):
        self.filename = filename
        self.train = train
        self.train_ratio = train_ratio
        self.seed = seed
        self._load_data()

    def _load_data(self):
        df = pd.read_csv(self.filename)
        if self.train:
            self.data = df.sample(
                frac=self.train_ratio, random_state=self.seed
            )
        else:
            self.data = df.sample(
                frac=1 - self.train_ratio, random_state=self.seed
            )

        self.targets = self.data["label"].unique()

    @property
    def n_classes(self) -> Union[float, int]:
        return len(self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        io_time = self.data.iloc[idx, 0].astype(np.float32)
        label = self.data.iloc[idx, 1]
        return io_time, label




class RegressionDataset(BaseDataset):
    TRAIN_RATIO = 0.8

    data: pd.DataFrame

    FEATURE_COLUMNS = [
        "fs_ave_oss_cpu",
        "fs_ave_mds_cpu",
        "fs_tot_gibs_read",
        "fs_tot_gibs_written",
        "jobsdb_concurrent_jobs",
        "fshealth_ost_avg_full_pct",
        "fs_tot_mkdir_ops",
        "fs_tot_rename_ops",
        "fs_tot_rmdir_ops",
        "fs_tot_unlink_ops",
        "APP_name_dbscan",
        "APP_name_hacc",
        "APP_name_ior",
        "APP_name_vpicio",
        "darshan_read_or_write_job_read",
        "darshan_read_or_write_job_write",
        "darshan_fpp_or_ssf_job_fpp",
        "darshan_fpp_or_ssf_job_shared",
    ]

    def __init__(
        self,
        filename: Union[str, Path],
        train_ratio: float = TRAIN_RATIO,
        train: bool = False,
    ):
        self._filename = filename
        self._n_features = len(self.FEATURE_COLUMNS)
        self._train = train
        self._train_ratio = train_ratio
        self.data = pd.read_csv(self._filename)
        self._load_data()

    def _load_data(self):
        # regression data, split evenly by `label`
        self.x, self.targets = split_evenly_by_classes(
            self.data,
            class_label="label",
            train=self._train,
            train_ratio=self._train_ratio,
        )

        log.info("Class distributions:")
        log.info(self.x.groupby(["label"]).size())

        self.y = self.x[
            "darshan_agg_perf_by_slowest_posix_gibs"
        ].reset_index(drop=True)
        self.x = self.x[self.FEATURE_COLUMNS].reset_index(drop=True)
        self.targets = self.targets.to_numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # log.info(
        #     f"IDX: {idx} CLS: {self.targets[idx]} Y: {self.y.iloc[idx].astype(np.float32)}"
        # )
        return (
            self.x.iloc[idx].astype(np.float32).values,
            self.targets[idx],
            self.y.iloc[idx].astype(np.float32),
        )


class APP_MATCHER(Dataset):
    FEATURE_COLUMNS = [
        "fs_ave_oss_cpu",
        "fs_ave_mds_cpu",
        "fs_tot_gibs_read",
        "fs_tot_gibs_written",
        "jobsdb_concurrent_jobs",
        "fshealth_ost_avg_full_pct",
        "fs_tot_mkdir_ops",
        "fs_tot_rename_ops",
        "fs_tot_rmdir_ops",
        "fs_tot_unlink_ops",
        "APP_name_dbscan",
        "APP_name_hacc",
        "APP_name_ior",
        "APP_name_vpicio",
        "darshan_read_or_write_job_read",
        "darshan_read_or_write_job_write",
        "darshan_fpp_or_ssf_job_fpp",
        "darshan_fpp_or_ssf_job_shared",
    ]

    def __init__(
        self,
        filename: Union[str, Path],
        train_ratio: float = 0.8,
        train: bool = False,
    ):
        super(APP_MATCHER, self).__init__()
        self._filename = filename
        self._n_features = len(self.FEATURE_COLUMNS)
        self._train = train
        self._train_ratio = train_ratio
        self.data = pd.read_csv(self._filename)
        self._load_data()

    def _load_data(self):
        # regression data, split evenly by `label`
        self.x, self.targets = split_evenly_by_classes(
            self.data,
            class_label="label",
            train=self._train,
            train_ratio=self._train_ratio,
        )


        self.y = self.x[
            "darshan_agg_perf_by_slowest_posix_gibs"
        ].reset_index(drop=True)
        self.x = self.x[self.FEATURE_COLUMNS].reset_index(drop=True)
        self.targets = self.targets.to_numpy()
        #print(self.y.shape)
        self.y=np.array(self.y)
        y=self.y.reshape((self.y.shape[0],1))
        #print(y.shape)
        self.features=np.concatenate((self.x,y),axis=1)
        self.features=self.features.astype('float32')
        #print(self.features.shape)
        self.grouped_examples={}
        for i in range(4):
            self.grouped_examples[i]=np.where((self.targets==i))[0]
        #print(self.group)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # log.info(
        #     f"IDX: {idx} CLS: {self.targets[idx]} Y: {self.y.iloc[idx].astype(np.float32)}"
        # )
        # pick some random class for the first image
        selected_class = random.randint(2, 3)

        # pick a random index for the first image in the grouped indices based of the label
        # of the class
        random_index_1 = random.randint(0, self.grouped_examples[selected_class].shape[0]-1)
        
        # pick the index to get the first image
        index_1 = self.grouped_examples[selected_class][random_index_1]

        # get the first image
        feature1 = torch.from_numpy(self.features[index_1])

        # same class
        if index % 2 == 0:
            # pick a random index for the second image
            random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0]-1)
            
            # ensure that the index of the second image isn't the same as the first image
            while random_index_2 == random_index_1:
                random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0]-1)
            
            # pick the index to get the second image
            index_2 = self.grouped_examples[selected_class][random_index_2]

            # get the second image
            feature2 = torch.from_numpy(self.features[index_2])

            # set the label for this example to be positive (1)
            target = torch.tensor(1, dtype=torch.float)
        
        # different class
        else:
            # pick a random class
            other_selected_class = random.randint(2, 3)

            # ensure that the class of the second image isn't the same as the first image
            while other_selected_class == selected_class:
                other_selected_class = random.randint(2, 3)

            
            # pick a random index for the second image in the grouped indices based of the label
            # of the class
            random_index_2 = random.randint(0, self.grouped_examples[other_selected_class].shape[0]-1)

            # pick the index to get the second image
            index_2 = self.grouped_examples[other_selected_class][random_index_2]

            # get the second image
            feature2 = torch.from_numpy(self.features[index_2])
            # set the label for this example to be negative (0)
            target = torch.tensor(0, dtype=torch.float)
        
        return feature1, feature2, target


        

'''
datasets=APP_MATCHER_2('data.csv')
train_loader = torch.utils.data.DataLoader(datasets,batch_size=1)
batch = next(iter(train_loader))

    
print(batch[0].shape)
print(batch[1].shape)
print(batch[2].shape)
sys.exit()
'''





__all__ = [
    "ClassificationDataset",
    "RegressionDataset",
    "RegressionAsClassificationDataset",
    "RegressionAsClassificationDatasetV2",
    "BaseDataset"
]

