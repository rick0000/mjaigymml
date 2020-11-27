from typing import Dict
from pathlib import Path

import yaml


class ExtractConfig:
    """
    特徴量抽出クラスを指定するコンフィグファイルのフォーマット定義
    特徴量抽出クラスは mjaigym_ml/features/custom フォルダ以下に作成する。

    on_reach_dahai: リーチまたは打牌時の特徴量抽出に使用する特徴量抽出クラス。
                    keyにファイル名、valueにクラス名を記載する。

    on_pon_chi_kan: ポン、チー、カン時の特徴量抽出に使用する特徴量抽出クラス。
                    NOTE:on_reach_dahai + on_pon_chi_kan を抽出器として使用するため、
                    on_reach_dahaiに含まれているものはon_pon_chi_kanに追加不要。
    """

    on_reach_dahai: Dict[str, str]
    on_pon_chi_kan: Dict[str, str]

    def __init__(self, config):
        if "on_reach_dahai" not in config:
            raise Exception("key on_reach_dahai not found")
        if "on_pon_chi_kan" not in config:
            raise Exception("key on_pon_chi_kan not found")

        self.on_reach_dahai = config["on_reach_dahai"]
        self.on_pon_chi_kan = config["on_pon_chi_kan"]

    def save(self, path: Path):
        output_dic = {
            "on_reach_dahai": self.on_reach_dahai,
            "on_pon_chi_kan": self.on_pon_chi_kan,
        }
        with open(path, "wt") as f:
            yaml.dump(output_dic, f)

    @classmethod
    def load(cls, path: Path):
        with open(path, "rt") as f:
            dic = yaml.safe_load(f)
        return ExtractConfig(dic)


if __name__ == "__main__":
    config = {
        "on_reach_dahai": {},
        "on_pon_chi_kan": {},
    }
    config = ExtractConfig(config)
    config.save("extract_config.yml")
    loaded_config = config.load("extract_config.yml")
    print(loaded_config)
