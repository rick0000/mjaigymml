from typing import List


class GrpDataset:
    """
    dataset for grobal reward predictor.
    input data scores index0 is seat0 score, output feature scores index0 is oya score.
    """

    def __init__(
        self,
        before_scores,
        end_scores,
        label_scores,
        honba=0,
        chicha=0,
        oya=0,
        kyotaku=0,
    ):
        self.honba = honba
        self.chicha = chicha
        self.oya = oya
        self.kyotaku = kyotaku
        self.before_scores = before_scores
        self.end_scores = end_scores
        self.label_scores = label_scores

        # oyaがindex0になるように並び替え
        self.oya_oriented_label_scores = [
            self.label_scores[self.oya],
            self.label_scores[self.shimocha],
            self.label_scores[self.toimen],
            self.label_scores[self.kamicha],
        ]

        # 同点時の席優先度。小さいほど有利。
        seat_priority = [0]*4
        for offset in range(4):
            # 起家が0、ラス親が3になる。
            seat_priority[(self.chicha+offset) % 4] = offset

        self.oya_oriented_seat_priority = [
            seat_priority[self.oya],
            seat_priority[self.shimocha],
            seat_priority[self.toimen],
            seat_priority[self.kamicha],
        ]

    @property
    def shimocha(self):
        return (self.oya+1) % 4

    @property
    def toimen(self):
        return (self.oya+2) % 4

    @property
    def kamicha(self):
        return (self.oya+3) % 4

    @property
    def oya_oriented_ranks(self):
        # oya, shimocha, toimen, kamichaの順位を返す
        return self._get_ranks(
            self.oya_oriented_label_scores,
            self.oya_oriented_seat_priority
        )

    @property
    def label_class(self):
        # oya, shimocha, toimen, kamichaの順位は24パターンに分類できるため
        # 0~15のクラス番号を返す

        return self._get_class_label(self.oya_oriented_ranks)

    @property
    def feature(self):
        return {
            "honba": self.honba,
            "kyotaku": self.kyotaku,
            "oya": self.oya,
            "before_score_oya": self.before_scores[self.oya],
            "before_score_shimocha": self.before_scores[self.shimocha],
            "before_score_toimen": self.before_scores[self.toimen],
            "before_score_kamicha": self.before_scores[self.kamicha],
            "end_score_oya": self.before_scores[self.oya],
            "end_score_shimocha": self.end_scores[self.shimocha],
            "end_score_toimen": self.end_scores[self.toimen],
            "end_score_kamicha": self.end_scores[self.kamicha],
        }

    def _get_ranks(self, scores, priority):
        """
        点数リストをもとに順位を計算する。
        出力の値は 0:1st, 1: 2nd, 2: 3rd, 3: 4th
        ---
        input:
            [10000, 20000, 50000, 30000]

        output:
            [3, 2, 0, 1]
        """

        rank_score_priority_sorted = sorted(
            zip(range(4), scores, priority),
            key=lambda x: (-x[1], x[2]),  # 得点が大きい、席優先度が小さい順に並ぶ。
        )
        return [rank for (rank, score, priority) in rank_score_priority_sorted]

    def _get_class_label(self, ranks):
        """
        4人の順位のリストを24クラスに変換する
        ---
        input:
            [3, 2, 0, 1]

        output:
            14
        """
        rank_label = _ranks.index(ranks)
        return rank_label

    def probs_to_each_ranks(self, class_probs):
        """
        24パターンで表されている予測順位確率を
        oya, shimocha, toimen, kamicha ごとの順位確率に変換する

        input:
            class_probs: List[float]      # len(prob) == 24

        output:
        {
            0: [rank1, rank2, rank3, rank4]     # oya
            1: [rank1, rank2, rank3, rank4]     # shimocha
            2: [rank1, rank2, rank3, rank4]     # toimen
            3: [rank1, rank2, rank3, rank4]     # kamicha
        }
        """
        oya_oriented_rank_probs = [
            [0]*4,  # oya
            [0]*4,  # shimocha
            [0]*4,  # toimen
            [0]*4,  # kamicha
        ]
        for class_ranks, class_prob in zip(_ranks, class_probs):
            # print(class_ranks, class_prob)
            for player in range(4):
                player_rank = class_ranks[player]
                oya_oriented_rank_probs[player][player_rank] += class_prob

        return oya_oriented_rank_probs


_ranks = [
    [0, 1, 2, 3],
    [0, 1, 3, 2],
    [0, 2, 1, 3],
    [0, 2, 3, 1],
    [0, 3, 1, 2],
    [0, 3, 2, 1],

    [1, 0, 2, 3],
    [1, 0, 3, 2],
    [1, 2, 0, 3],
    [1, 2, 3, 0],
    [1, 3, 0, 2],
    [1, 3, 2, 0],

    [2, 0, 1, 3],
    [2, 0, 3, 1],
    [2, 1, 0, 3],
    [2, 1, 3, 0],
    [2, 3, 0, 1],
    [2, 3, 1, 0],

    [3, 0, 2, 1],
    [3, 0, 1, 2],
    [3, 1, 0, 2],
    [3, 1, 2, 0],
    [3, 2, 0, 1],
    [3, 2, 1, 0],
]
