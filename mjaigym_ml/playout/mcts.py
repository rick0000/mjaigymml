from mjaigym.board.archive_board import ArchiveBoard


class Node(object):
    def __init__(self, board: ArchiveBoard, depth: int):
        self.board = board
        self.visit_count = 0

    def playout(self):
        """

        """
        pass

    def get_next_mcts_action_candidates(self):
        """
        打牌後はツモまたは副露アクションを返す。
        ツモまたは副露後は打牌アクションを返す。
        """
        pass


# プレイアウト機能に必要な要件
# アクション生成フェーズが必要ってこった。
# だから、archive_boardが適任だと思う
