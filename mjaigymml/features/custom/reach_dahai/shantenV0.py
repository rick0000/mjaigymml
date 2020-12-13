import numpy as np

from mjaigymml.features.custom.feature_reach_dahai import FeatureReachDahai
from mjaigym.board import BoardState
from mjaigym.board.function.rs_shanten_analysis import RsShantenAnalysis


class ShantenV0(FeatureReachDahai):

    def __init__(self):
        self.shanten_analysis = RsShantenAnalysis()

    def get_length(cls) -> int:
        return 4  # shanten 0, 1, 2, over3

    def calc(self, result: np.array, board_state: BoardState, player_id: int):
        player_tehais = board_state.tehais[player_id]
        nums = [0] * 34
        for pai in player_tehais:
            nums[pai.id] += 1
        furo_num = len(board_state.furos[player_id])
        shanten = self.shanten_analysis.calc_shanten(
            nums, furo_num
        )
        shanten = max(0, shanten)
        shanten = min(3, shanten)
        result[shanten, :] = 1
