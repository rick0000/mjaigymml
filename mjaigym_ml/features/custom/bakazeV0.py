import numpy as np

from mjaigym_ml.features.custom.feature import Feature
from mjaigym.board import BoardState


class BakazeV0(Feature):

    def get_length(cls) -> int:
        return 1

    def calc(cls, result: np.array, board_state: BoardState, player_id: int):
        for anpai in board_state.anpais[player_id]:
            result[0, anpai.pai_id] = 1
