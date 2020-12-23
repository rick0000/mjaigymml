import numpy as np

from mjaigymml.features.custom.feature_common import FeatureCommon
from mjaigym.board import BoardState


class KyotakuV0(FeatureCommon):

    def get_length(cls) -> int:
        return 10

    def calc(self, result: np.array, board_state: BoardState):
        kyotaku_num = min(10, board_state.kyotaku)
        if kyotaku_num == 0:
            return
        else:
            target_ch = kyotaku_num - 1
            result[target_ch, :] = 1
