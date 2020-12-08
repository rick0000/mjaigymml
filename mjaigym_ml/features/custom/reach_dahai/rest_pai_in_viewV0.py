import numpy as np

from mjaigym.board.mj_move import MjMove
from mjaigym_ml.features.custom.feature_reach_dahai import FeatureReachDahai
from mjaigym.board import BoardState


class RestPaiInViewV0(FeatureReachDahai):

    def get_length(cls) -> int:
        return 4

    def calc(self, result: np.array, board_state: BoardState, player_id: int):
        rests = board_state.restpai_in_view[player_id]
        for i, num in enumerate(rests):
            result[:num, i] = 1
