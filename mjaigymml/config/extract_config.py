from typing import Dict
from pathlib import Path
from dataclasses import dataclass

import yaml

from mjaigymml.config.config_base import ConfigBase


@dataclass
class ExtractConfig(ConfigBase):
    """
    特徴量抽出クラスを指定するコンフィグファイルのフォーマット定義
    特徴量抽出クラスは mjaigymml/features/custom フォルダ以下に作成する。

    common:
        ドラ、山残り枚数など全プレーヤー共通の特徴量。
        keyにファイル名、valueにクラス名を記載する。

    on_reach_dahai:
        リーチまたは打牌時の特徴量抽出に使用する特徴量抽出クラス。
        keyにファイル名、valueにクラス名を記載する。

    on_pon_chi_kan:
        ポン、チー、カン時の特徴量抽出に使用する特徴量抽出クラス。
        keyにファイル名、valueにクラス名を記載する。
        NOTE:on_reach_dahai + on_pon_chi_kan を抽出器として使用するため、
        on_reach_dahaiに含まれているものはon_pon_chi_kanに追加不要。
    """
    common: Dict[str, str]
    on_reach_dahai: Dict[str, str]
    on_reach_dahai_oracle: Dict[str, str]
    on_pon_chi_kan: Dict[str, str]

    def __init__(self, config):
        if "common" not in config:
            raise Exception("key common not found")
        if "on_reach_dahai" not in config:
            raise Exception("key on_reach_dahai not found")
        if "on_reach_dahai_oracle" not in config:
            raise Exception("key on_reach_dahai_oracle not found")
        if "on_pon_chi_kan" not in config:
            raise Exception("key on_pon_chi_kan not found")

        self.common = config["common"]
        self.on_reach_dahai = config["on_reach_dahai"]
        self.on_reach_dahai_oracle = config["on_reach_dahai_oracle"]
        self.on_pon_chi_kan = config["on_pon_chi_kan"]

    def save(self, path: Path):
        output_dic = {
            "common": self.common,
            "on_reach_dahai": self.on_reach_dahai,
            "on_reach_dahai_oracle": self.on_reach_dahai_oracle,
            "on_pon_chi_kan": self.on_pon_chi_kan,
        }
        with open(path, "wt") as f:
            yaml.dump(output_dic, f)


if __name__ == "__main__":
    import os
    fpath = "extract_config.yml"
    if os.path.isfile(fpath):
        config = ExtractConfig.load(fpath)
    else:
        config = {
            "common": {},
            "on_reach_dahai": {},
            "on_reach_dahai_oracle": {},
            "on_pon_chi_kan": {},
        }
        config = ExtractConfig(config)

    config.save("extract_config.yml")
    loaded_config = config.load("extract_config.yml")
    print(loaded_config)
