from mjaigym.board.function.pai import Pai
from mjaigym.board.function.yaku_name import YAKU_CHANNEL_MAP
# from mjaigym.board.function.dfs_result import DfsResult
from mjaigym.board.function.efficient_dfs import Dfs
from mjaigym.board.function.rs_shanten_analysis import RsShantenAnalysis
from mjaigym.board import BoardState
from mjaigym_ml.features.custom.feature_reach_dahai import FeatureReachDahai
import numpy as np


class HorapointDfsV0(FeatureReachDahai):

    target_points = [
        1000,
        1300,
        1600,
        2000,
        2400,
        2600,
        3200,
        3900,
        4800,
        5200,
        7700,
        8000,
        12000,
        16000,
        18000,
        24000,
        32000,
        36000,
        48000,
    ]
    DEPTH = 2

    def __init__(self):
        self.dfs = Dfs()
        self.shanten_analysis = RsShantenAnalysis()

    def get_length(cls) -> int:
        yaku_ch = len(YAKU_CHANNEL_MAP) * cls.DEPTH
        point_ch = len(cls.target_points) * cls.DEPTH
        return yaku_ch + point_ch

    def calc(self, result: np.array, board_state: BoardState, player_id: int):

        player_tehai = board_state.tehais[player_id]

        nums = [0] * 34
        for t in player_tehai:
            nums[t.id] += 1
        tehai_akadora_num = len([p for p in player_tehai if p.is_red])

        player_furos = board_state.furos[player_id]

        # num 14 check
        if len(player_tehai) + len(player_furos) * 3 != 14:
            # return
            pass

        furo_akadora_num = 0
        for furo in player_furos:
            furo_akadora_num += len([p for p in furo.pais if p.is_red])

        # # ignore -1, more than 2

        oya = board_state.oya == player_id
        bakaze = board_state.bakaze
        jikaze = board_state.jikaze[player_id]
        doras = [p.succ for p in Pai.from_list(board_state.dora_markers)]
        uradoras = []
        num_akadoras = tehai_akadora_num + furo_akadora_num

        shanten_normal, shanten_kokushi, shanten_chitoitsu = \
            self.shanten_analysis.calc_all_shanten(nums, len(player_furos))

        hora_results = []
        if 0 <= shanten_normal <= self.DEPTH-1:
            normal_results = self.dfs.dfs_with_score_normal(
                nums,
                player_furos,
                self.DEPTH,
                oya=oya,
                bakaze=bakaze,
                jikaze=jikaze,
                doras=doras,
                uradoras=uradoras,
                num_akadoras=num_akadoras,
                shanten_normal=shanten_normal,
            )
            hora_results.extend(normal_results)

        if 0 <= shanten_chitoitsu <= self.DEPTH-1:
            chitoitsu_results = self.dfs.dfs_with_score_chitoitsu(
                nums,
                player_furos,
                self.DEPTH,
                oya=oya,
                bakaze=bakaze,
                jikaze=jikaze,
                doras=doras,
                uradoras=uradoras,
                num_akadoras=num_akadoras,
                shanten_chitoitsu=shanten_chitoitsu,
            )
            hora_results.extend(chitoitsu_results)

        if 0 <= shanten_kokushi <= self.DEPTH-1:
            kokushi_results = self.dfs.dfs_with_score_kokushi(
                nums,
                player_furos,
                self.DEPTH,
                oya=oya,
                shanten_kokushi=shanten_kokushi,
            )
            hora_results.extend(kokushi_results)

        hora_results = [r for r in hora_results if r.valid()]

        if len(hora_results) == 0:
            return

        for hora_result in hora_results:
            # あるpaiを打牌した際にN手先である役が成立する場合に1を立てる。

            distance = hora_result.hora_path_length
            start_index = (distance-1) * len(YAKU_CHANNEL_MAP)
            print(distance)
            for yaku, fan in hora_result.get_yakus():
                if yaku == "dora":
                    fan = fan if fan < 12 else 12
                    yaku = f"{yaku}{fan}"

                if yaku not in YAKU_CHANNEL_MAP:
                    continue

                target_index = YAKU_CHANNEL_MAP[yaku]
                for dahai_id in hora_result.dahaiable_ids:
                    result[start_index + target_index, dahai_id] = 1

            # あるpaiを打牌した際にN手先にX点得られる場合に1を立てる。
            point_feature_start_index = len(YAKU_CHANNEL_MAP) * self.DEPTH
            start_index = (distance-1) * len(self.target_points) + \
                point_feature_start_index

            for point_index, target_point in enumerate(self.target_points):
                if target_point < hora_result.get_point():
                    for dahai_id in hora_result.dahaiable_ids:
                        result[start_index + point_index, dahai_id] = 1
