import numpy as np

from mjaigym_ml.features.custom.feature import Feature
from mjaigym.board import BoardState


class DiscardV0(Feature):

    def get_length(cls) -> int:
        return 24

    def calc(cls, result: np.array, board_state: BoardState, player_id: int):
        for i, pai in enumerate(board_state.sutehais[player_id]):
            result[i, pai.id] = 1
