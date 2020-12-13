from typing import Dict
from pathlib import Path
from dataclasses import dataclass

import yaml

from mjaigymml.config.config_base import ConfigBase


@dataclass
class ModelConfig(ConfigBase):
    """
    学習用パラメータを指定するコンフィグファイルのフォーマット定義

    resnet_repeat: int
        Residual層の繰り返し数

    mid_channels: int
        フィルタ数

    learning_rate: float
        学習率

    batch_size: int
        バッチサイズ
    """
    resnet_repeat: int = 20
    mid_channels: int = 128
    learning_rate: float = 0.0001
    batch_size: int = 128

    def __init__(self, config):
        if "resnet_repeat" not in config:
            raise Exception("key resnet_repeat not found")
        if "mid_channels" not in config:
            raise Exception("key mid_channels not found")
        if "learning_rate" not in config:
            raise Exception("key learning_rate not found")
        if "batch_size" not in config:
            raise Exception("key batch_size not found")

        self.resnet_repeat = config["resnet_repeat"]
        self.mid_channels = config["mid_channels"]
        self.learning_rate = config["learning_rate"]
        self.batch_size = config["batch_size"]

    def save(self, path: Path):
        output_dic = {
            "resnet_repeat": self.resnet_repeat,
            "mid_channels": self.mid_channels,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
        }
        with open(path, "wt") as f:
            yaml.dump(output_dic, f)


if __name__ == "__main__":
    config = {
        "resnet_repeat": 20,
        "mid_channels": 256,
        "learning_rate": 0.01,
        "batch_size": 256,
    }
    config = ModelConfig(config)
    config.save("model_config.yml")
    loaded_config = config.load("model_config.yml")
    print(loaded_config)
