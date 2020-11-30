import numpy as np

from mjaigym_ml.features.custom.feature_reach_dahai import FeatureReachDahai
from mjaigym.board import BoardState


class ChiV0(FeatureReachDahai):

    def get_length(cls) -> int:
        return 4

    def calc(cls, result: np.array, board_state: BoardState, player_id: int):
        counts = {}
        player_furos = board_state.furos[player_id]
        for furo in player_furos:
            if furo.pai_id not in counts:
                counts[furo.pai_id] = 0
            counts[furo.pai_id] += 1

        for pai_id, num in counts.items():
            result[0:num, pai_id] = 1
