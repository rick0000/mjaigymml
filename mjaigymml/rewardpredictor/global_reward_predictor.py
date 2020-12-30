from dataclasses import dataclass
from typing import List, Dict
from itertools import product
import pickle
import pprint

from sklearn.metrics import confusion_matrix
import pandas as pd

from mjaigymml.rewardpredictor.model import \
    Model, LogisticRegressionModel
from mjaigymml.rewardpredictor.grp_dataset import GrpDataset


@dataclass
class RankPredictResult:
    seat_rank_probs: Dict[int, List[float]]


class GlobalRewardPredictor:
    NEED_COLUMNS = [
        "honba",
        "kyotaku",
        "oya",
        "max_over_30000",
        "diff_0", "diff_1", "diff_2", "diff_3",
        "before_score_0",
        "before_score_01", "before_score_02", "before_score_03",
        "before_score_12", "before_score_13",
        "before_score_23",
    ]
    KYOKUS = list(product([1, 2, 3, 4], ("E", "S")))

    def __init__(self, model_cls: Model):
        self.models = {}
        for (kyoku, bakaze) in GlobalRewardPredictor.KYOKUS:
            self.models[(kyoku, bakaze)] = model_cls()

    def load(self, file_path) -> "GlobalRewardPredictor":
        with open(file_path, "rb") as f:
            self.models = pickle.load(f)

    def save(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self.models, f)

    def predict(
            self,
            before_scores,
            end_scores,
            kyoku,
            bakaze,
            honba,
            chicha,
            oya,
            kyotaku,
            is_tonnnan
    ) -> List[RankPredictResult]:

        # if tonpu, use S1, S2, S3, S4
        if not is_tonnnan:
            bakaze = "S"

        data = GrpDataset(
            before_scores=before_scores,
            end_scores=end_scores,
            label_scores=[0, 0, 0, 0],
            kyoku=kyoku,
            bakaze=bakaze,
            honba=honba,
            chicha=chicha,
            oya=oya,
            kyotaku=kyotaku
        )
        feature = data.feature
        df = pd.DataFrame(
            data=[feature], columns=GlobalRewardPredictor.NEED_COLUMNS)
        preds = self.models[(kyoku, bakaze)].predict_proba(df.values)

        rank_probs = \
            [GrpDataset.probs_to_each_ranks(p) for p in preds]

        # convert from oya index0 to seat0 index0
        results = []
        for rank_prob in rank_probs:
            record_results = {}
            for i in range(4):
                record_results[i] = rank_prob[i]

            results.append(RankPredictResult(record_results))
        return results

    def train(self, df):
        for c in self.NEED_COLUMNS:
            assert c in df.columns, f"{c} not found"

        # remove tonpu
        df = df[df["is_tonnnan"]]

        for (kyoku, bakaze) in GlobalRewardPredictor.KYOKUS:
            print("start train", bakaze, kyoku)
            kyoku_df = df[(df["kyoku"] == kyoku)
                          & (df["bakaze"] == bakaze)]
            kyoku_df = kyoku_df.reset_index(drop=True)
            print("kyoku_df.shape", kyoku_df.shape)
            label = kyoku_df["label_class"]
            feature = kyoku_df[self.NEED_COLUMNS]
            self.models[(kyoku, bakaze)].fit(feature, label)

    def evaluate(self, df):
        for c in self.NEED_COLUMNS:
            assert c in df.columns

        # remove tonpu
        df = df[df["is_tonnnan"]]

        for (kyoku, bakaze) in GlobalRewardPredictor.KYOKUS:
            print("start evaluate", bakaze, kyoku)
            kyoku_df = df[(df["kyoku"] == kyoku)
                          & (df["bakaze"] == bakaze)]
            kyoku_df = kyoku_df.reset_index(drop=True)
            print("kyoku_df.shape", kyoku_df.shape)
            label = kyoku_df["label_class"]
            feature = kyoku_df[self.NEED_COLUMNS]
            result = self.models[(kyoku, bakaze)].predict(feature)
            c_mat = confusion_matrix(label, result)
            pprint.pprint(c_mat)
            pd.DataFrame(c_mat).to_csv(f"eval_{bakaze}_{kyoku}.csv")


if __name__ == "__main__":
    import pandas as pd

    use_class = LogisticRegressionModel
    class_name = use_class().__class__.__name__
    grp = GlobalRewardPredictor(use_class)
    train = False
    evaluate = False
    if train:
        df = pd.read_pickle("output/dataset/train_grp_dataset.pkl")

        grp.train(df)
        grp.save(f"grp_{class_name}.pkl")

    grp.load(f"grp_{class_name}.pkl")

    if evaluate:
        df = pd.read_pickle("output/dataset/test_grp_dataset.pkl")

        grp.evaluate(df)

    kyoku = 4
    bakaze = "S"
    before_scores = [25000, 25000, 25000, 25000]
    # end_scores = [11000, 25000, 39000, 24000]
    end_scores = [25000, 25000, 24000, 26000]
    print(bakaze, kyoku, before_scores, "->", end_scores)
    preds = grp.predict(
        before_scores=before_scores,
        end_scores=end_scores,
        kyoku=kyoku,
        bakaze=bakaze,
        honba=0,
        chicha=0,
        oya=3,
        kyotaku=3,
        is_tonnnan=True
    )
    print(preds)
