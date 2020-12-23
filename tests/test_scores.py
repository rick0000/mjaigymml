import random

import numpy as np

import conftest
from mjaigym.board.function.pai import Pai
from mjaigym.board.function.yaku_name import YAKU_CHANNEL_MAP
# from mjaigym.board.function.dfs_result import DfsResult
from mjaigym.board.function.efficient_dfs import Dfs
from mjaigym.board.function.rs_shanten_analysis import RsShantenAnalysis
from mjaigym.board import Board
from mjaigym.board import BoardState
from mjaigymml.features.custom.feature_reach_dahai import FeatureReachDahai


from mjaigymml.features.custom.reach_dahai.score_diffV0 import ScoreDiffV0


def test_scores():
    board = Board()

    state = board.get_state()
    state.state["scores"] = [0, -1000, 1000, -65000]
    player_id = 0
    print(state)
    calclator = ScoreDiffV0()
    result = np.zeros((ScoreDiffV0.get_length(), 34))
    calclator.calc(result, state, player_id)
    show_result = result[:, 0].flatten()
    print(show_result[0:12])
    print(show_result[12:24])
    print(show_result[24:36])
    print(show_result[36:48])

    assert list(show_result[0:12]) == [
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  # 1位との点差
    assert list(show_result[12:24]) == [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 2位との点差
    assert list(show_result[24:36]) == [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # 3位との点差
    assert list(show_result[36:48]) == [
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # 4位との点差


if __name__ == "__main__":
    test_scores()
