from typing import Dict
from pathlib import Path
from dataclasses import dataclass

import yaml

from mjaigymml.config.config_base import ConfigBase


@dataclass
class TrainConfig(ConfigBase):
    """
    教師あり学習用パラメータを指定するコンフィグファイルのフォーマット定義

    model_type: str [dahai, reach, pon, kan, chi]
        学習するモデルの種類

    sampling_rate: float (0.0, 1.0]
        学習時に使用する局面数のサンプリング割合。
        1.0の場合は使用可能な全局面を使って学習を行う。

    use_oracle: bool
        Trueの場合Oracle特徴量使って教師あり学習を行う
        打牌のみ挙動が変わる。
    """
    model_type: str
    sampling_rate: float
    use_oracle: bool

    def __init__(self, config):
        if "model_type" not in config:
            raise Exception("key model_type not found")
        if "sampling_rate" not in config:
            raise Exception("key sampling_rate not found")
        if "use_oracle" not in config:
            raise Exception("key use_oracle not found")

        self.model_type = config["model_type"]
        self.sampling_rate = config["sampling_rate"]
        self.use_oracle = config["use_oracle"]

        assert 0 < self.sampling_rate <= 1.0, \
            f"sampling_rate must be in (0.0, 1.0], input:{self.sampling_rate}"

        valid_type = ["dahai", "reach", "pon", "kan", "chi"]
        assert self.model_type in valid_type, \
            f"model_type must be in {valid_type}, input:{self.model_type}"

    def save(self, path: Path):
        output_dic = {
            "model_type": self.model_type,
            "sampling_rate": self.sampling_rate,
            "use_oracle": self.use_oracle,
        }
        with open(path, "wt") as f:
            yaml.dump(output_dic, f)


if __name__ == "__main__":
    config = {
        "model_type": "dahai",
        "sampling_rate": 0.05,
        "use_oracle": False,
    }
    config = TrainConfig(config)
    config.save("train_config.yml")
    loaded_config = config.load("train_config.yml")
    print(loaded_config)
