from dataclasses import dataclass
from typing import List
import pickle

from sklearn.linear_model import LinearRegression


@dataclass
class PredictResult:
    player_id: int
    rank_probs: List[float]  # { 1st, 2nd, 3rd, 4th probs }


class GlobalRewardPredictor:
    def __init__(self):
        self.models = {}

    @classmethod
    def load(cls, file_path) -> "GlobalRewardPredictor":
        pass

    def save(self, file_path):
        pass

    def predict(self, df) -> List[PredictResult]:
        pass

    def train(self, df):
        need_columns = []
        for c in need_columns:
            assert c in df.columns
        pass

    def _train_one_kyoku(self, df):
        pass

    def change_seat0_index0_to_oya_index0(
            self,
            scores: List[int],
            oya: int
    ) -> List[int]:
        assert 0 <= oya <= 3
        return scores[oya:] + scores[:oya]

    def change_oya_index0_to_seat0_index0(
            self,
            preds: List[PredictResult],
            oya: int
    ) -> List[PredictResult]:
        assert 0 <= oya <= 3
        return preds[-oya:] + preds[:-oya]


class LinearModel:

    @classmethod
    def load(cls, file_path) -> "LinearModel":
        pass

    def save(self, file_path):
        pass

    def fit(self, label, feature):
        pass

    def predict(self, feature):
        pass


class MLPModel:
    @classmethod
    def load(cls, file_path) -> "MLPModel":
        pass

    def save(self, file_path):
        pass

    def fit(self, label, feature):
        pass

    def predict(self, feature):
        pass
