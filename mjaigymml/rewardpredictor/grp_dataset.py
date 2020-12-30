from typing import List
from collections import ChainMap


DIV = 1


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
        kyoku=1,
        bakaze="E",
        honba=0,
        chicha=0,
        oya=0,
        kyotaku=0,
    ):
        self.kyoku = kyoku
        self.bakaze = bakaze
        self.honba = honba
        self.chicha = chicha
        self.oya = oya
        self.kyotaku = kyotaku
        self._before_scores = before_scores
        self._end_scores = end_scores
        self._diff = [e - b for (e, b) in zip(end_scores, before_scores)]
        self._raw_label_scores = label_scores

        # # oyaがindex0になるように並び替え
        # self.oya_oriented_label_scores = \
        #     self.get_oya_oriented(self._raw_label_scores)
        # self.oya_oriented_before_scores = \
        #     self.get_oya_oriented(self._before_scores)
        # self.oya_oriented_end_scores = \
        #     self.get_oya_oriented(self._end_scores)
        # self.oya_oriented_diff = \
        #     self.get_oya_oriented(self._diff)

        # 同点時の席優先度。小さいほど有利。
        self.seat_priority = [0]*4
        # for offset in range(4):
        #     # 起家が0、ラス親が3になる。
        #     seat_priority[(self.chicha+offset) % 4] = offset

        # self.oya_oriented_seat_priority = self.get_oya_oriented(seat_priority)

        self.ranks = self._get_ranks(
            self._raw_label_scores,
            self.seat_priority
        )

    # def get_oya_oriented(self, scores):
    #     return [
    #         scores[self.oya],
    #         scores[self.shimocha],
    #         scores[self.toimen],
    #         scores[self.kamicha],
    #     ]

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
    def labels(self):
        return ChainMap(
            self.label_class,
            self.label_ranks,
            self.label_scores
        )

    @property
    def label_class(self):
        # 順位は24パターンに分類できるため
        # 0~23のクラス番号を返す
        return {
            "label_class": self._get_class_label(self.ranks)
        }

    @property
    def label_scores(self):
        # 最終点数を返す
        return {
            "label_score_0": self._end_scores[0],
            "label_score_1": self._end_scores[1],
            "label_score_2": self._end_scores[2],
            "label_score_3": self._end_scores[3],
        }

    @property
    def label_ranks(self):
        # 最終順位を返す
        return {
            "label_rank_0": self.ranks[0],
            "label_rank_1": self.ranks[1],
            "label_rank_2": self.ranks[2],
            "label_rank_3": self.ranks[3],
        }

    @property
    def feature(self):

        return {
            "kyoku": self.kyoku,
            "bakaze": self.bakaze,
            "honba": self.honba*300/DIV,
            "kyotaku": self.kyotaku*1000/DIV,
            "oya": self.oya,
            "max_over_30000": max(self._end_scores) >= 30000,
            "diff_0": self._diff[0]/DIV,
            "diff_1": self._diff[1]/DIV,
            "diff_2": self._diff[2]/DIV,
            "diff_3": self._diff[3]/DIV,
            "before_score_0": self._before_scores[0]/DIV,
            "before_score_01":
                (self._before_scores[0] -
                 self._before_scores[1])/DIV,
            "before_score_02":
                (self._before_scores[0] -
                 self._before_scores[2])/DIV,
            "before_score_03":
                (self._before_scores[0] -
                 self._before_scores[3])/DIV,
            "before_score_12":
                (self._before_scores[1] -
                 self._before_scores[2])/DIV,
            "before_score_13":
                (self._before_scores[1] -
                 self._before_scores[3])/DIV,
            "before_score_23":
                (self._before_scores[2] -
                 self._before_scores[3])/DIV,
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
            [z for z in zip(range(4), scores, priority)
             ],  # seat, score, priority
            key=lambda x: (-x[1], x[2]),  # 得点が大きい、席優先度が小さい順に並ぶ。
        )

        seat_ranks = [rank for (rank, score, priority)
                      in rank_score_priority_sorted]

        ranks = [
            seat_ranks.index(0),  # 席が0の人のランク
            seat_ranks.index(1),
            seat_ranks.index(2),
            seat_ranks.index(3),
        ]

        return ranks

    def _get_class_label(self, ranks):
        """
        4人の順位のリストを24クラスに変換する
        ---
        input:
            [3, 2, 0, 1]

        output:
            14
        """
        rank_label = RANKS.index(ranks)
        return rank_label

    @classmethod
    def probs_to_each_ranks(cls, class_probs):
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
        for class_ranks, class_prob in zip(RANKS, class_probs):
            # print(class_ranks, class_prob)
            for player in range(4):
                player_rank = class_ranks[player]
                oya_oriented_rank_probs[player][player_rank] += class_prob

        return oya_oriented_rank_probs


RANKS = [
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

if __name__ == "__main__":
    gd = GrpDataset(
        [10000, 20000, 50000, 30000],
        [10000, 20000, 50000, 30000],
        [10000, 20000, 50000, 30000],
    )
    print(gd.ranks)
