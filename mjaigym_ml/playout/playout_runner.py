import enum
from typing import List, Dict
from dataclasses import dataclass

from mjaigym.board.board_state import BoardState


class PlayoutRunner():
    def __init__(self):
        pass

    def set_state_by_board_state(self, board_state):
        pass

    def set_state_by_actions(self, action_history: List[Dict]):
        pass

    def playout(self, num):
        pass


@dataclass
class PlayoutResult():
    diff0: int
    diff1: int
    diff2: int
    diff3: int
