from pathlib import Path
from typing import Tuple, Dict, Generator

import numpy as np
import pandas as pd
import h5py


class FeatureStorageHDF5():
    def __init__(self, dir_path):
        self.feature_file_path = Path(dir_path) / "feature.hdf5"
        self.label_dir_path = Path(dir_path) / "label"
        self.feature_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.label_dir_path.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(self.feature_file_path, "a")

    def save(self, fname, feature: Dict[str, np.array], label: pd.DataFrame):
        fname = Path(fname)
        label_path = self.label_dir_path / f"{fname}.csv.gz"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label.to_csv(label_path, index=False, compression='gzip')

        # save feature into hdf5
        for key, value in feature.items():
            hdf5_path = fname / "feature" / key
            result = self.f.get(str(hdf5_path), default=None)
            if result is None:
                self.f.create_dataset(
                    str(hdf5_path),
                    data=value,
                )

    def load(self, fname) -> Tuple[np.array, pd.DataFrame]:
        raise NotImplementedError()

    def load_all(self) -> Generator[None, Tuple[np.array, pd.DataFrame], None]:
        raise NotImplementedError()
