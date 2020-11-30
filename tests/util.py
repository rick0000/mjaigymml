import conftest
from mjaigym.board.function.pai import Pai
from mjaigym.board.function.yaku_name import YAKU_CHANNEL_MAP
# from mjaigym.board.function.dfs_result import DfsResult
from mjaigym.board.function.efficient_dfs import Dfs
from mjaigym.board.function.rs_shanten_analysis import RsShantenAnalysis
from mjaigym.board import Board
from mjaigym.board import BoardState
from mjaigym_ml.features.custom.feature_reach_dahai import FeatureReachDahai
import numpy as np
import random


def get_random_dahai_board():
    board = Board()

    # random step
    dahai_count = 0
    while True:
        state = board.get_state()
        if state.previous_action['type'] == "dahai":
            dahai_count += 1

        if dahai_count > 10 and state.previous_action['type'] == "tsumo":
            break

        actions = {}
        for i in range(4):
            actions[i] = random.choice(state.possible_actions[i])
        board.step(actions)

    return board
