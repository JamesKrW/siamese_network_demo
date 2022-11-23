# https://github.com/mit-ll-responsible-ai/hydra-zen-examples/blob/main/image_classifier/hydra_zen_example/image_classifier/utils.py
import random
from numbers import Number
from typing import Any, Sized, Tuple, cast

import torch
from torch.utils.data import random_split as _random_split
from torch.utils.data.dataset import Dataset

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split




def split_evenly_by_classes(
    data: pd.DataFrame,
    class_label: str,
    train: bool,
    train_ratio: float,
    shuffle: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        data,
        data[class_label],
        test_size=1 - train_ratio,
        stratify=data[class_label],
        shuffle=shuffle,
    )

    if train:
        return X_train, y_train

    return X_test, y_test


def random_split(
    dataset,
    val_split: float = 0.1,
    train: bool = True,
    random_seed: int = 32,
) -> Dataset:
    g = torch.Generator().manual_seed(random_seed)
    nval = int(len(cast(Sized, dataset)) * val_split)
    ntrain = len(cast(Sized, dataset)) - nval
    train_data, val_data = _random_split(dataset, [ntrain, nval], g)

    if hasattr(dataset, "targets"):
        cast(Any, train_data).targets = dataset.targets
        cast(Any, val_data).targets = dataset.targets

    if train:
        return train_data

    return val_data


def set_seed(random_seed: Number) -> None:
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)


__all__ = ["split_evenly_by_classes", "random_split", "set_seed"]
