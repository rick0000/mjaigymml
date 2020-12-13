import numpy as np

from mjaigymml.features.custom.feature_reach_dahai import FeatureReachDahai
from mjaigym.board import BoardState


class PonV0(FeatureReachDahai):

    def get_length(cls) -> int:
        return 1

    def calc(self, result: np.array, board_state: BoardState, player_id: int):
        player_furos = board_state.furos[player_id]
        for furo in player_furos:
            if furo.type == "pon":
                result[0, furo.pai_id] = 1
