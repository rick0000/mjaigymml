from typing import Dict

import numpy as np

from mjaigym_ml.features.custom.feature_pon_chi_kan import FeaturePonChiKan
from mjaigym.board.board_state import BoardState
from mjaigym.board.mj_move import MjMove
from mjaigym.board.function.pai import Pai


class FuroReddora(FeaturePonChiKan):
    """
    副露で開示されている赤ドラの枚数
    """
    @classmethod
    def get_length(cls):
        return 3

    def calc(
            cls,
            result: np.array,
            board_state: BoardState,
            player_id: int,
            candidate_furo: Dict):
        num = board_state.furo_open_red_dora_nums[player_id]
        result[0:num, :] = 1
