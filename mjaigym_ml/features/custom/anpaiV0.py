import numpy as np

from mjaigym_ml.features.custom.feature_reach_dahai import FeatureReachDahai
from mjaigym.board import BoardState


class AnpaiV0(FeatureReachDahai):

    def get_length(cls) -> int:
        return 1

    def calc(cls, result: np.array, board_state: BoardState, player_id: int):
        for anpai in board_state.anpais[player_id]:
            result[0, anpai.id] = 1
