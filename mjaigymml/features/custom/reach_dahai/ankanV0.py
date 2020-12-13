import numpy as np

from mjaigym.board.mj_move import MjMove
from mjaigymml.features.custom.feature_reach_dahai import FeatureReachDahai
from mjaigym.board import BoardState


class AnkanV0(FeatureReachDahai):

    def get_length(cls) -> int:
        return 1

    def calc(self, result: np.array, board_state: BoardState, player_id: int):
        for furo in board_state.furos[player_id]:
            if furo.type == MjMove.ankan.value:
                result[0, furo.pai_id] = 1
