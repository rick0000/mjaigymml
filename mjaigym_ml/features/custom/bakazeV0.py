from typing import List, Tuple
import enum
import numpy as np
import pprint
import os

from mjaigym.board.mj_move import MjMove
from mjaigym_ml.features.custom.feature import Feature
from mjaigym.board import BoardState
from mjaigym.board.function.pai import Pai

class BakazeV0(Feature):
    @abstractclassmethod
    def get_length(cls)->int:
        return 1
        
    @abstractclassmethod
    def calc(cls, result:np.array, board_state:BoardState, player_id:int):
        for anpai in board_state.anpais[player_id]:
            result[0, anpai.pai_id] = 1
