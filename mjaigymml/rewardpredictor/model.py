import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


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

    def get_model(self):
        raise NotImplementedError()


class LinearRegressionModel(Model):

    def get_model(self):
        return LinearRegression()


class LogisticRegressionModel(Model):

    def get_model(self):
        return LogisticRegression()
