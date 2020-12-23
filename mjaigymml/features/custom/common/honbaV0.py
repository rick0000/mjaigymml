import numpy as np

from mjaigymml.features.custom.feature_common import FeatureCommon
from mjaigym.board import BoardState


class HonbaV0(FeatureCommon):

    def get_length(cls) -> int:
        return 10

    def calc(self, result: np.array, board_state: BoardState):
        honba_num = min(10, board_state.honba)
        if honba_num == 0:
            return
        else:
            target_ch = honba_num - 1
            result[target_ch, :] = 1
