from typing import Tuple, Generator
import enum
import numpy as np
import pandas as pd

from .feature_storage_hdf5 import FeatureStorageHDF5
from .feature_storage_localfs import FeatureStorageLocalFs


class StorageType(enum.Enum):
    hdf5 = "hdf5"
    localfs = "localfs"


class FeatureStorage():
    VALID_STORAGES = [
        StorageType.hdf5.value,
        StorageType.localfs.value,
    ]

    def __init__(self, storage_type="hdf5"):
        if storage_type not in self.VALID_STORAGES:
            raise Exception(
                "inputed storage_type not found. valid targets:"
                + f"{self.VALID_STORAGES}"
            )
        if storage_type == StorageType.hdf5.value:
            self.storage = FeatureStorageHDF5()
        elif storage_type == StorageType.localfs.value:
            self.storage = FeatureStorageLocalFs()

    def save(self, fname, feature: np.array, label: pd.DataFrame):
        self.storage.save(fname, feature, label)

    def load(self, fname) -> Tuple[np.array, pd.DataFrame]:
        return self.storage.load(fname)

    def load_all(self) -> Generator[None, Tuple[np.array, pd.DataFrame], None]:
        return self.storage.load_all()
