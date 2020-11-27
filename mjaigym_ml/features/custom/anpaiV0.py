from typing import List, Tuple
import enum
import numpy as np
import pprint
import os

from mjaigym.board.mj_move import MjMove
from mjaigym_ml.features.custom.feature_reach_dahai import FeatureReachDahai
from mjaigym.board import BoardState
from mjaigym.board.function.pai import Pai

class AnpaiV0(FeatureReachDahai):
    def get_length(cls)->int:
        return 1
    
    def calc(cls, result:np.array, board_state:BoardState, player_id:int):
        for anpai in board_state.anpais[player_id]:
            result[0, anpai.id] = 1
