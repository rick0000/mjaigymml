from abc import abstractmethod, abstractclassmethod
import numpy as np
from .feature_reach_dahai import FeatureReachDahai
from mjaigym.board.board_state import BoardState


class AAAv0(FeatureReachDahai):
    @classmethod
    def get_length(cls):
        return 1

    def calc(cls, result: np.array, board_state: BoardState, player_id: int):
        pass
