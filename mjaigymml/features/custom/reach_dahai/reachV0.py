import numpy as np

from mjaigym.board.mj_move import MjMove
from mjaigymml.features.custom.feature_reach_dahai import FeatureReachDahai
from mjaigym.board import BoardState


class ReachV0(FeatureReachDahai):

    def get_length(cls) -> int:
        return 1

    def calc(self, result: np.array, board_state: BoardState, player_id: int):
        if board_state.reach[player_id]:
            result[0, :] = 1
