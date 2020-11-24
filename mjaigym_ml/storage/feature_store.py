from abc import ABCMeta, abstructmethod
from typing import Tuple, Generator

import numpy as np
import pandas as pd

class FeatureStorage(ABCMeta):
    
    @abstructmethod
    def save(self, fname, feature:np.array, label:pd.DataFrame):
        raise NotImplementedError()

    @abstructmethod
    def load(self, fname)->Tuple[np.array, pd.DataFrame]:
        raise NotImplementedError()

    @abstructmethod
    def load_all(self)->Generator[None, Tuple[np.array, pd.DataFrame], None]:
        raise NotImplementedError()
