from typing import Dict

import numpy as np

from mjaigym_ml.features.custom.feature_pon_chi_kan import FeaturePonChiKan
from mjaigym.board.board_state import BoardState
from mjaigym.board.mj_move import MjMove
from mjaigym.board.function.pai import Pai


class FuroCandidateV0(FeaturePonChiKan):
    """
    盤面状態に対して候補手になっているポン、チー、カンアクションを表す。
    """
    @classmethod
    def get_length(cls):
        return 5

    def calc(
            cls,
            result: np.array,
            board_state: BoardState,
            player_id: int,
            candidate_furo: Dict):
        target_index = None
        if candidate_furo["type"] == MjMove.chi.value:
            target_index = 0
        elif candidate_furo["type"] == MjMove.pon.value:
            target_index = 1
        elif candidate_furo["type"] in \
            [MjMove.kakan.value,
                MjMove.daiminkan.value,
                MjMove.ankan.value]:
            target_index = 2
        else:
            raise Exception("not intended path")
        result[target_index, :] = 1

        pais = []
        if candidate_furo["type"] == MjMove.ankan.value:
            pais += candidate_furo["consumed"]
        else:
            pais += ([candidate_furo["pai"]] + candidate_furo["consumed"])
        pais = Pai.from_list(pais)

        min_pai_id = min([pai.id for pai in pais])
        result[3, min_pai_id] = 1

        contains_red = any([pai.is_red for pai in pais])
        if contains_red:
            result[4, :] = 1
