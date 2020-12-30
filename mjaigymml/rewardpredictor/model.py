import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class Model(object):
    """
    モデル抽象化用クラス
    基本実装としてsklearn用のロジックを記載している
    """

    def __init__(self):
        self.model = self.get_model()

    @classmethod
    def load(cls, file_path: Path) -> "Model":
        model = cls()
        model.model = pickle.load(file_path)
        return model

    def save(self, file_path: Path):
        pickle.dump(self.model, file_path)

    def fit(self, feature: np.array, label: np.array):
        self.model.fit(feature, label)

    def predict(self, feature: np.array):
        return self.model.predict(feature)

    def predict_proba(self, feature: np.array):
        return self.model.predict_proba(feature)

    def get_model(self):
        raise NotImplementedError()


class LinearRegressionModel(Model):

    def get_model(self):
        return LinearRegression()


class LogisticRegressionModel(Model):

    def get_model(self):
        return LogisticRegression(max_iter=1000)


class LGBModel(Model):
    def fit(self, feature: np.array, label: np.array):
        import lightgbm as lgb
        sampled_index = np.random.choice(
            np.array(range(len(feature))), min(100000, len(feature)))
        train, val = train_test_split(sampled_index, test_size=0.33)
        sampled_feature = feature.iloc[train]
        sampled_label = label.iloc[train]
        sampled_feature_val = feature.iloc[val]
        sampled_label_val = label.iloc[val]

        lgb_train = lgb.Dataset(sampled_feature, sampled_label)
        lgb_eval = lgb.Dataset(sampled_feature_val,
                               sampled_label_val, reference=lgb_train)
        params = {
            "objective": "multiclass",
            "num_class": 24,
            "verbose": 0,
        }
        self.model = lgb.train(
            params=params,
            train_set=lgb_train,
            num_boost_round=500,
            early_stopping_rounds=10,
            valid_sets=lgb_eval,
            # verbose_eval=False,
        )

    def predict(self, feature: np.array):
        return np.argmax(self.model.predict(feature), axis=1)

    def predict_proba(self, feature: np.array):
        return self.model.predict(feature)

    def get_model(self):
        return None
