"""
盤面状態をもとに合法手の中からアクションを返す
"""
import random
from typing import Dict

from mjaigym.board.board_state import BoardState


class AgentFacede():
    def __init__(self, id=None):
        self.id = id

    def think(self, board_state:BoardState)->Dict:
        if "id" in board_state.previous_action:
            self.id = board_state.previous_action["id"]

        my_possible_actions = board_state.possible_actions[self.id]
        random_selected_action = random.choice(my_possible_actions)
        return random_selected_action
