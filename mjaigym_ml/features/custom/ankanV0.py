from typing import List, Tuple
import enum
import numpy as np
import pprint
import os

from mjaigym.board.mj_move import MjMove
from mjaigym_ml.features.custom.feature_reach_dahai import FeatureReachDahai
from mjaigym.board import BoardState
from mjaigym.board.function.pai import Pai

class AnkanV0(FeatureReachDahai):
    
    def get_length(cls)->int:
        return 1
        
    
    def calc(cls, result:np.array, board_state:BoardState, player_id:int):
        for furo in board_state.furos[player_id]:
            if furo.type == MjMove.ankan.value:
                result[0, furo.pai_id] = 1
