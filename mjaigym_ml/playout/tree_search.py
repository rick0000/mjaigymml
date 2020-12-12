from collections import deque
import copy

from mjaigym.board.function.pai import Pai
from mjaigym.board.archive_board import ArchiveBoard
from mjaigym.board.board import Board


class Node(object):
    def __init__(self, board: ArchiveBoard, depth_count: int = 0):
        self.board = board
        self.depth_count = depth_count

    @classmethod
    def extract(cls, root_node: "Node", depth: int):
        """
        全探索を行う
        """
        node_queue = deque()
        assert root_node.board.previous_action["type"] == "tsumo"
        root_actor = root_node.board.previous_action["actor"]
        node_queue.append(root_node)
        node_count = 0
        while len(node_queue) > 0:
            node_count += 1
            node = node_queue.popleft()
            if node.depth_count >= depth:
                continue

            next_actions = node.get_next_action_candidates()
            for action in next_actions:
                # print(action)
                board = ArchiveBoard(scene_mjsons=node.board.dealer_history)
                board.step(action)
                # print("steped, previous_action", board.previous_action)
                new_depth_count = node.depth_count
                # and action["actor"] == root_actor:
                if action["type"] == "tsumo":
                    new_depth_count += 1

                new_node = Node(board, new_depth_count)
                node_queue.append(new_node)
        print("node evaluate count", node_count)

    def get_next_action_candidates(self):
        """
        打牌後はツモまたは副露アクションを返す。
        ツモまたは副露後は打牌アクションを返す。
        カンは無視する。
        """
        candidates = []

        if self.board.previous_action['type'] == "dahai":
            actor = (self.board.previous_action["actor"] + 1) % 4
            # add tsumo candidates

            for pai_str, num in self.board.yama.rest_paistr_nums.items():
                if num == 0:
                    continue

                candidates.append({
                    "type": "tsumo",
                    "actor": actor,
                    "pai": pai_str,
                })

            # add furo candidates
            for player_id, actions in self.board.possible_actions.items():
                for action in actions:
                    if action["type"] != "none":
                        candidates.append(action)
            pass
        elif self.board.previous_action['type'] == "tsumo" or\
                self.board.previous_action['type'] == "reach":
            # dahai or reach
            for player_id, actions in self.board.possible_actions.items():
                for action in actions:
                    if action["type"] != "none":
                        candidates.append(action)

        return candidates


def playout(root_board: Board, playout_num: int):
    import random
    for _ in range(playout_num):
        board = copy.deepcopy(root_board)
        while not board.is_end:
            actions = {
                0: random.choice(board.possible_actions[0]),
                1: random.choice(board.possible_actions[1]),
                2: random.choice(board.possible_actions[2]),
                3: random.choice(board.possible_actions[3]),
            }
            board.step(actions)
        print(board.dealer_history[-3:])


if __name__ == "__main__":
    # 局開始状態をランダムに生成
    from mjaigym.board.board import Board

    initial_board = Board()
    initial_board.reset()  # start_game
    initial_board.step(
        dict(zip(range(4), [{"type": "none"}]*4)))  # start_kyoku
    initial_board.step(
        dict(zip(range(4), [{"type": "none"}]*4)))  # oya tsumo
    board_state = initial_board.get_state()

    extract_root_board = ArchiveBoard(initial_board.dealer_history)

    root_node = Node(extract_root_board)
    # Node.extract(root_node, 1)
    playout(initial_board, 100)
