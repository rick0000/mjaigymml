from abc import abstractmethod, abstractclassmethod
import numpy as np
from .feature import Feature
from mjaigym.board.board_state import BoardState

class FeatureAAAv0(Feature):
    @classmethod
    def get_length(cls):
        return 1
    
    def calc(cls, result:np.array, board_state:BoardState, player_id:int):
        pass