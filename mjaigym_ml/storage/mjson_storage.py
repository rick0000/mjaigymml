from pathlib import Path
from typing import Generator
from mjaigym.mjson import Mjson

class MjsonStorage():

    def get_mjsons()->Generator[None,Mjson,None]:
        raise NotImplementedError()
