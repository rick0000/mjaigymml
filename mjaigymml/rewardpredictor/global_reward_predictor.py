from dataclasses import dataclass
from typing import List
from itertools import product

from mjaigymml.rewardpredictor.model import Model, LogisticRegressionModel
from mjaigymml.rewardpredictor.grp_dataset import GrpDataset


@dataclass
class PredictResult:
    player_id: int
    rank_probs: List[float]  # { 1st, 2nd, 3rd, 4th probs }
    class_probs: List[float]  # { 16 class probs }

    def calc_rank_probs(self, seat):
        pass


class GlobalRewardPredictor:
    NEED_COLUMNS = ["honba", "oya", "kyotaku", "before_scores", "end_scores"]
    kyokus = list(product([1, 2, 3, 4], ("E", "S")))

    def __init__(self, model_cls: Model):
        self.models = {}
        for (kyoku_id, bakaze) in self.kyokus:
            self.models[(kyoku_id, bakaze)] = model_cls()

    @classmethod
    def load(cls, file_path) -> "GlobalRewardPredictor":
        pass

    def save(self, file_path):
        pass

    def predict(self, df) -> List[PredictResult]:
        pass

    def train(self, df):
        for c in self.NEED_COLUMNS:
            assert c in df.columns

        for (kyoku_id, bakaze) in self.kyokus:
            kyoku_df = df[(df["kyoku_id"] == kyoku_id)
                          & (df["bakaze"] == bakaze)]
            label = kyoku_df["target_label"]
            feature = kyoku_df[self.NEED_COLUMNS]
            self.models[(kyoku_id, bakaze)].fit(feature, label)

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


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_pickle("output/dataset/test_grp_dataset.pkl")

    grp = GlobalRewardPredictor(LogisticRegressionModel)
    grp.train(df)
