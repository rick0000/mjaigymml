from pathlib import Path
from typing import Tuple, Dict, Generator

import numpy as np
import pandas as pd


class FeatureStorageLocalFs():
    def __init__(self, dir_path):
        self.feature_dir_path = Path(dir_path) / "feature"
        self.feature_dir_path.mkdir(parents=True, exist_ok=True)
        self.label_dir_path = Path(dir_path) / "label"
        self.label_dir_path.mkdir(parents=True, exist_ok=True)

    def save(self, fname, feature: Dict[str, np.array], label: pd.DataFrame):
        fname = Path(fname)
        label_path = self.label_dir_path / f"{fname}.csv.gz"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label.to_csv(label_path, index=False, compression='gzip')

        # save feature into hdf5
        feature_file_path = self.feature_dir_path / f"{fname}.npz"
        np.savez_compressed(
            feature_file_path,
            **feature,
        )

    def load(self, fname) -> Tuple[np.array, pd.DataFrame]:
        raise NotImplementedError()

    def load_all(self) -> Generator[None, Tuple[np.array, pd.DataFrame], None]:
        raise NotImplementedError()
