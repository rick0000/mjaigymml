from typing import List
from dataclasses import dataclass


@dataclass
class FeatureScript:
    """
    特徴量抽出クラスのファイルパス
    """
    file_path: str


@dataclass
class ModelScript:
    """
    機械学習モデル構造定義クラスのファイルパス
    """
    file_path: str


@dataclass
class ModelParams:
    """
    機械学習モデル構造定義クラスのファイルパス
    """
    mid_channels: int
    resblock_repeat: int
    learning_rate: float


@dataclass
class ObserversConfig:
    """
    特徴量抽出に使用する特徴量抽出用クラス群の定義
    """
    file_path: List[FeatureScript]


@dataclass
class ModelConfig:
    """
    学習に使用するモデル定義クラスとパラメータ
    """
    model_script: ModelScript
    model_params: ModelParams


@dataclass
class ExperimentConfig:
    """
    """
    # Modelコンフィグ
    observer_config: ObserversConfig
    model_configs: ModelConfig
