from pathlib import Path
from typing import Generator
from mjaigym.mjson import Mjson
from .mjson_storage import MjsonStorage

class LocalFileMjsonStorage(MjsonStorage):
    def __init__(self, input_dir, max_num=-1):
        self.input_dir = input_dir
        self.max_num = max_num

    def get_mjsons(self)->Generator[None,Mjson,None]:
        for index, mjson in enumerate(Path(self.input_dir).glob("**/*.mjson")):
            if self.max_num != -1 and index >= self.max_num:
                break 
            yield Mjson.load(mjson)
            
