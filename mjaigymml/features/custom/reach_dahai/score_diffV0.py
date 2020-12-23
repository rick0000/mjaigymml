import numpy as np

from mjaigymml.features.custom.feature_reach_dahai import FeatureReachDahai
from mjaigym.board import BoardState


class ScoreDiffV0(FeatureReachDahai):
    """
    ある順位のプレーヤーと対象プレーヤーの得点差を1000点刻みで表現する。
    0点から63000点までを6ビット=6チャネルを使って表現する。
    点差が正の場合は前半6チャネル、負の場合は後半6チャネルを使用する。
    """
    RANGE_ONSESSIDE_LENGTH = 6
    SCORE_RANGE_LENGTH = RANGE_ONSESSIDE_LENGTH * 2
    FORMAT_STRING = "{:0" + str(RANGE_ONSESSIDE_LENGTH) + "b}"

    @classmethod
    def get_length(cls) -> int:
        return cls.SCORE_RANGE_LENGTH * 4

    def calc(self, result: np.array, board_state: BoardState, player_id: int):
        scores = board_state.scores
        sorted_scores = sorted(scores, reverse=True)

        player_score = board_state.scores[player_id]
        for i, score in enumerate(sorted_scores):

            diff = player_score - score
            diff = min(63000, max(-63000, diff))  # -63000 ~ 63000
            diff_scaled = int(diff / 1000)  # -63 ~ 63

            if diff_scaled >= 0:
                binarized_diff = self.FORMAT_STRING.format(diff_scaled)
                offset = 6
            else:
                binarized_diff = self.FORMAT_STRING.format(-diff_scaled)
                offset = 0
            start = self.SCORE_RANGE_LENGTH * i + offset

            for j, num in enumerate(binarized_diff):
                result[start+j, :] = int(num)
