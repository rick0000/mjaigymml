from pathlib import Path

import yaml
import json


class ConfigBase(object):
    @classmethod
    def load(cls, path: Path):
        with open(path, "rt") as f:
            dic = yaml.safe_load(f)
        return cls(dic)

    def get_dict(self):
        return self.__dict__

    def get_json(self):
        return json.dumps(self.__dict__)
