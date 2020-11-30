import numpy as np
from mjaigym_ml.features.custom.feature_pon_chi_kan import FeaturePonChiKan
from mjaigym.board.board_state import BoardState


class BBBV0(FeaturePonChiKan):
    @classmethod
    def get_length(cls):
        return 1

    def calc(cls, result: np.array, board_state: BoardState, player_id: int):
        pass
