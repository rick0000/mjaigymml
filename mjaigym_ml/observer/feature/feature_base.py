from abc import ABCMeta

import numpy as np

from mjaigym.board.board_state import BoardState

class FeatureBase(ABCMeta):
    def calc(board_state:BoardState)->np.array:
        raise NotImplementedError()
