from typing import Dict
from pathlib import Path

import yaml

class ExtractorConfig:
    on_reach_dahai:Dict[str,str]
    on_pon_chi_kan:Dict[str,str]

    def __init__(self, config):
        if "on_reach_dahai" not in config:
            raise Exception("key on_reach_dahai not found")
        if "on_pon_chi_kan" not in config:
            raise Exception("key on_pon_chi_kan not found")

        self.on_reach_dahai = config["on_reach_dahai"]
        self.on_pon_chi_kan = config["on_pon_chi_kan"]

    def save(self, path:Path):
        output_dic = {
            "on_reach_dahai":self.on_reach_dahai,
            "on_pon_chi_kan":self.on_pon_chi_kan,
        }
        with open(path, "wt") as f:
            yaml.dump(output_dic, f)

    def load(self, path:Path):
        with open(path, "rt") as f:
            dic = yaml.safe_load(f)
        return dic


if __name__ == "__main__":
    config = {
        "on_reach_dahai":{},
        "on_pon_chi_kan":{},
    }
    config = ExtractorConfig(config)
    config.save("extract_config.yml")
    loaded_config = config.load("extract_config.yml")
    print(loaded_config)