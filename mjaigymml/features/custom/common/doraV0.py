import numpy as np

from mjaigymml.features.custom.feature_common import FeatureCommon
from mjaigym.board import BoardState
from mjaigym.board.function.pai import Pai


class DoraV0(FeatureCommon):

    def get_length(cls) -> int:
        return 4

    def calc(self, result: np.array, board_state: BoardState):
        dora_markers = board_state.dora_markers
        nums = {}
        for dora_maker in dora_markers:
            pai = Pai.from_str(dora_maker)
            dora_pai = pai.succ
            if dora_pai.id not in nums:
                nums[dora_pai.id] = 0
            nums[dora_pai.id] += 1

        for pai_id, n in nums.items():
            if n > 0:
                result[0:n, pai_id] = 1
