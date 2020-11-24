from abc import ABCMeta, abstructmethod
from typing import Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from .feature_store import FeatureStorage

class LocalFileFeatureStorage(FeatureStorage):
    def __init__(self, dir_path):
        self.dir_path = Path(dir_path)

    @abstructmethod
    def save(self, fname:Path, feature:np.array, label:pd.DataFrame):
        feature_path, label_path = _get_feature_label_paths(fname)
        np.save(feature_path, feature)
        label.to_csv(label_path)

    @abstructmethod
    def load(self, fname:Path)->Tuple[np.array, pd.DataFrame]:
        feature_path, label_path = _get_feature_label_paths(fname)
        feature = np.load(feature_path)
        label = pd.read_csv(label_path)
        return feature, label

    @abstructmethod
    def load_all(self):
        label_paths = self.dir_path.grep("label.csv")
        for label_path in label_paths:
            fname = label_path.name.replace("label.csv", "")
            feature, label = self.load(fname)
            yield feature, label

    def _get_feature_label_paths(self, fname):
        fpath = self.dir_path / fname + "feature.npy"
        lpath = self.dir_path / fname + "label.csv"
        return fpath, lpath
        

