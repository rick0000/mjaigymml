from typing import List
from dataclasses import dataclass


@dataclass
class ObserverScript:
    """
    特徴量抽出クラスを指定する
    """
    file_path: str


@dataclass
class InferenceConfig:
    """
    推論時用コンフィグ
    特徴量抽出関数とモデル定義関数を記載する
    """
    observer_scripts: List[ObserverScript]
    model_script: str


@dataclass
class SupervisedLearningTrainConfig:
    """
    教師あり学習用コンフィグ
    特徴量抽出関数とモデル定義関数を記載する
    """
    observer_scripts: List[ObserverScript]
    model_script: str

    def dump_inference_config(self) -> InferenceConfig:
        """
        推論時に必要なコンフィグを返す
        """
        return InferenceConfig()


@dataclass
class ReinforcementLearningTrainConfig:
    """
    強化学習用コンフィグ
    特徴量抽出関数とモデル定義関数を記載する
    """
    observer_scripts: List[ObserverScript]
    model_script: str

    def dump_inference_config(self):
        """
        推論時に必要なコンフィグを出力する
        """
