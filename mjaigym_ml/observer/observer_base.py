"""
特徴抽出に使用するFeatureクラスを列挙する。
"""
from abc import ABCMeta

import numpy as np

from mjaigym.board.board_state import BoardState


class ObserverBase(ABCMeta):
    def observe(self, board_state: BoardState) -> np.array:
        return np.zeros((1, 34, 1))
