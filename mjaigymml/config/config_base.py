from pathlib import Path

import yaml


class ConfigBase(object):
    @classmethod
    def load(cls, path: Path):
        with open(path, "rt") as f:
            dic = yaml.safe_load(f)
        return cls(dic)
