from typing import Dict, List
import importlib
import random
from pathlib import Path

import numpy as np

from mjaigym.mjson import Mjson
from mjaigym.board.board_state import BoardState
from mjaigym.board.archive_board import ArchiveBoard
from mjaigymml.config.extract_config import ExtractConfig
from mjaigymml.config.train_config import TrainConfig
from mjaigymml.features.custom.feature_common import FeatureCommon
from mjaigymml.features.custom.feature_reach_dahai import FeatureReachDahai
from mjaigymml.features.custom.feature_pon_chi_kan import FeaturePonChiKan
from mjaigymml.features.feature_analysis \
    import LabelRecord, FeatureRecord, Dataset


class FeatureAnalyser():
    """
    ゲーム解析情報（mjsonクラス）をもとに特徴量を生成するためのクラス
    """

    def __init__(self, extract_config: ExtractConfig):
        self.extract_config = extract_config
        self.common_extractors = []
        self.reach_dahai_extractors = []
        self.pon_chi_kan_extractors = []

        # configで指定された特徴量抽出クラスをimportする
        # 共通特徴量のimport
        for module_name, target_class_name in \
                self.extract_config.common.items():
            target_class = self._get_module_class(
                "common",
                module_name,
                target_class_name
            )
            class_instance = target_class()
            assert isinstance(class_instance, FeatureCommon)
            self.common_extractors.append(class_instance)

        # 全アクション用特徴量のimport
        for module_name, target_class_name in \
                self.extract_config.on_reach_dahai.items():
            target_class = self._get_module_class(
                "reach_dahai",
                module_name,
                target_class_name)
            class_instance = target_class()
            assert isinstance(class_instance, FeatureReachDahai)
            self.reach_dahai_extractors.append(class_instance)

        # ポンチーカン用特徴量のimport
        for module_name, target_class_name in \
                self.extract_config.on_pon_chi_kan.items():
            target_class = self._get_module_class(
                "pon_chi_kan",
                module_name,
                target_class_name)
            class_instance = target_class()
            assert isinstance(class_instance, FeaturePonChiKan)
            self.pon_chi_kan_extractors.append(class_instance)

        self.common_length = sum(
            [f.get_length() for f in self.common_extractors])
        self.reach_dahai_length = sum(
            [f.get_length() for f in self.reach_dahai_extractors])
        self.pon_chi_kan_length = sum(
            [f.get_length() for f in self.pon_chi_kan_extractors])

    def get_reach_dahai_feature_length(self):
        return self.common_length\
            + self.reach_dahai_length * 4

    def get_pon_chi_kan_feature_length(self):
        return self.common_length\
            + self.reach_dahai_length * 4\
            + self.pon_chi_kan_length * 4

    def reset_extractor_state(self):
        """
        キャッシュなど内部状態のリセット
        """
        for extractor in self.common_extractors:
            extractor.reset()
        for extractor in self.reach_dahai_extractors:
            extractor.reset()
        for extractor in self.pon_chi_kan_extractors:
            extractor.reset()

    def analyze_list(self, mjson_list: List[Dict]) -> Dataset:
        """
        辞書のリストで与えられたアクションの履歴について特徴量の算出を行う。
        全アクションではなく最後の状態についてのみ特徴量の算出を行う。
        """
        board = ArchiveBoard()
        datasets = []
        # 最終行の盤面状態作成
        for kyoku_line_index, action in enumerate(mjson_list):
            board.step(action)

        board_state = board.get_state()
        dahai, reach, pon, chi, kan = self._possible_action_types(
                board_state, None)

        start_kyoku_line = mjson_list[0]
        assert start_kyoku_line["type"] == "start_kyoku"

        label_record = LabelRecord(
            filename="",
            mjson_line_index=len(mjson_list)-1,
            kyoku_line_index=len(mjson_list)-1,
            kyoku_line_num=len(mjson_list),
            kyoku_index=start_kyoku_line["kyoku"],
            kyoku=start_kyoku_line["kyoku"],
            bakaze=start_kyoku_line["bakaze"],
            honba=start_kyoku_line["honba"],
            kyotaku=start_kyoku_line["kyotaku"],
            dahai=dahai,
            reach=reach,
            chi=chi,
            pon=pon,
            kan=kan,

            score_diff_0=0,
            score_diff_1=0,
            score_diff_2=0,
            score_diff_3=0,

            initial_score_0=25000,
            initial_score_1=25000,
            initial_score_2=25000,
            initial_score_3=25000,

            end_score_0=25000,
            end_score_1=25000,
            end_score_2=25000,
            end_score_3=25000,

            next_action_type=None, 
            next_action=None,
            candidate_action_type=None,  # 打牌では利用しないカラム
            next_and_candidate_is_same=None,  # 打牌では利用しないカラム
            candidate_action=None,  # 打牌では利用しないカラム
        )

        if pon or chi or kan:
            # TODO:LabelRecordに追加
            # candidate_action_type=None,  # 打牌では利用しないカラム
            # next_and_candidate_is_same=None,  # 打牌では利用しないカラム
            # candidate_action=None,  # 打牌では利用しないカラム
            for player_id in range(4):
                for possible_action in board_state.possible_actions[player_id]:
                    if possible_action['type'] in ["pon", "chi", "ankan", "kakan", "daiminkan"]:
                        # 1副露アクション候補に対して1レコード発生
                        dataset = Dataset(
                            label_record, board_state, possible_action)
                        datasets.append(dataset)
        else:
            # 1行に対して1レコード発生
            dataset = Dataset(label_record, board_state)
            datasets.append(dataset)

        return datasets

    def analyse_mjson(
        self,
        mjson,
    ) -> List[Dataset]:
        """
        Mjsonオブジェクトで与えられたアクションの履歴についてラベルと盤面状態の算出を行う。
        """
        if isinstance(mjson, Path):
            mjson = Mjson.load(mjson)

        board = ArchiveBoard()
        datasets = []

        line_count = 0
        for kyoku_index, kyoku in enumerate(mjson.game.kyokus):
            for kyoku_line_index, action in enumerate(kyoku.kyoku_mjsons):
                board.step(action)
                board_state = board.get_state()

                # ファイル上での行数通し番号
                mjson_line_index = line_count + kyoku_line_index
                # それぞれのモデルの学習対象局面でない行はスキップ
                if not self._is_label_scene(board_state):
                    continue

                if kyoku_line_index+1 >= len(kyoku.kyoku_mjsons):
                    continue

                # 次のアクションを取得
                next_action = kyoku.kyoku_mjsons[kyoku_line_index+1]

                # どのモデルの学習対象かチェック
                dahai, reach, pon, chi, kan = self._possible_action_types(
                    board_state, next_action)

                # 抽出対象でない場合はスキップ
                if not (dahai or reach or pon or chi or kan):
                    continue

                label_record = LabelRecord(
                    filename=mjson.path.name,
                    mjson_line_index=mjson_line_index,
                    kyoku_line_index=kyoku_line_index,
                    kyoku_line_num=len(kyoku.kyoku_mjsons),
                    kyoku_index=kyoku_index,
                    kyoku=kyoku.kyoku_id,
                    bakaze=kyoku.bakaze,
                    honba=kyoku.honba,
                    kyotaku=board_state.kyotaku,
                    dahai=dahai,
                    reach=reach,
                    chi=chi,
                    pon=pon,
                    kan=kan,

                    score_diff_0=kyoku.result_scores[0] -
                    kyoku.initial_scores[0],
                    score_diff_1=kyoku.result_scores[1] -
                    kyoku.initial_scores[1],
                    score_diff_2=kyoku.result_scores[2] -
                    kyoku.initial_scores[2],
                    score_diff_3=kyoku.result_scores[3] -
                    kyoku.initial_scores[3],

                    initial_score_0=kyoku.initial_scores[0],
                    initial_score_1=kyoku.initial_scores[1],
                    initial_score_2=kyoku.initial_scores[2],
                    initial_score_3=kyoku.initial_scores[3],

                    end_score_0=kyoku.result_scores[0],
                    end_score_1=kyoku.result_scores[1],
                    end_score_2=kyoku.result_scores[2],
                    end_score_3=kyoku.result_scores[3],

                    next_action_type=next_action["type"],
                    next_action=next_action,
                    candidate_action_type=None,  # 打牌では利用しないカラム
                    next_and_candidate_is_same=None,  # 打牌では利用しないカラム
                    candidate_action=None,  # 打牌では利用しないカラム
                )

                if pon or chi or kan:
                    for player_id in range(4):
                        for possible_action in board_state.possible_actions[player_id]:
                            if possible_action['type'] in ["pon", "chi", "ankan", "kakan", "daiminkan"]:
                                # 1副露アクション候補に対して1レコード発生
                                dataset = Dataset(
                                    label_record, board_state, possible_action)
                                datasets.append(dataset)
                else:
                    # 1行に対して1レコード発生
                    dataset = Dataset(label_record, board_state)
                    datasets.append(dataset)

            line_count += len(kyoku.kyoku_mjsons)

        return datasets

    def filter_datasets(
            self,
            datasets: List[Dataset],
            train_config: TrainConfig
    ):
        """
        train_config.model_type で指定されたラベルに関するデータを返す。
        train_config.sampling_rate で指定された割合でダウンサンプリングを行う。
        """

        # get type records
        if train_config.model_type == "dahai":
            records = [d for d in datasets if d.label.dahai]
        elif train_config.model_type == "reach":
            records = [d for d in datasets if d.label.reach]
        elif train_config.model_type == "pon":
            records = [d for d in datasets if d.label.pon]
        elif train_config.model_type == "chi":
            records = [d for d in datasets if d.label.chi]
        elif train_config.model_type == "kan":
            records = [d for d in datasets if d.label.kan]
        else:
            raise Exception("not intended path.")

        if len(records) == 0:
            return []

        # sampling
        take_num = max(1, int(len(records) * train_config.sampling_rate))
        records = random.sample(records, take_num)

        return records

    def calc_feature(self, datasets: List[Dataset]):
        common_result_array = np.zeros(
            (len(datasets), self.common_length, 34))

        reach_dahai_result_array = np.zeros(
            (len(datasets), 4, self.reach_dahai_length, 34))

        for dataset_index, dataset in enumerate(datasets):
            common_start_index = 0
            reach_dahai_start_index = 0
            # calc common feature
            for e in self.common_extractors:
                target_array = common_result_array[
                    dataset_index,
                    common_start_index:common_start_index + e.get_length(),
                    :]
                e.calc(
                    result=target_array,
                    board_state=dataset.board_state,
                )
                common_start_index += e.get_length()

            # calc reach dahai feature
            for e in self.reach_dahai_extractors:
                for player_id in range(4):
                    target_array = reach_dahai_result_array[
                        dataset_index,
                        player_id,
                        reach_dahai_start_index:reach_dahai_start_index + e.get_length(),
                        :]
                    e.calc(
                        result=target_array,
                        board_state=dataset.board_state,
                        player_id=player_id,
                    )
                reach_dahai_start_index += e.get_length()

            pon_chi_kan_result_array = None
            # calc pon_chi_kan_feaature
            if dataset.label.pon or dataset.label.chi or dataset.label.kan:
                # 出現する場合のみメモリ確保
                pon_chi_kan_result_array = np.zeros(
                    (self.pon_chi_kan_length, 34))
                pon_chi_kan_index = 0
                for e in self.pon_chi_kan_extractors:
                    target_array = pon_chi_kan_result_array[
                        pon_chi_kan_index:pon_chi_kan_index + e.get_length(),
                        :]

                    e.calc(
                        result=target_array,
                        board_state=dataset.board_state,
                        player_id=dataset.candidate_furo_action['actor'],
                        candidate_furo_action=dataset.candidate_furo_action
                    )
                    pon_chi_kan_index += e.get_length()

            # update result object
            feature = FeatureRecord(
                common_result_array[dataset_index],
                reach_dahai_result_array[dataset_index],
                pon_chi_kan_result_array
            )
            dataset.set_feature(feature)

    def _is_label_scene(self, state: BoardState):
        # 選択可能なアクションが複数ない場合
        # =Noneアクションしか出来ない
        # =教師データに不適当
        if all([len(actions) == 1 for actions in
                state.possible_actions.values()]):
            return False
        return True

    def _possible_action_types(self, state: BoardState, next_action: Dict):
        dahai = False
        reach = False
        pon = False
        chi = False
        kan = False

        if next_action:
            if next_action["type"] == "dahai" and \
                    (not state.reach[next_action["actor"]]):
                dahai = True
        else:
            if state.previous_action["type"] == "dahai" and \
                    (not state.reach[next_action["actor"]]):
                dahai = True
                
        # other check
        for player_actions in state.possible_actions.values():
            for action in player_actions:
                if action["type"] == "reach":
                    reach = True
                elif action["type"] == "pon":
                    pon = True
                elif action["type"] == "chi":
                    chi = True
                elif action["type"] == "kan":
                    kan = True

        return dahai, reach, pon, chi, kan

    def _get_module_class(self, dir_name, module_name, target_class_name):
        target_module_name = self._filter_dotpy(module_name)
        target_module = importlib.import_module(
            f"mjaigymml.features.custom.{dir_name}.{target_module_name}")
        target_class = getattr(target_module, target_class_name)
        return target_class

    def _filter_dotpy(self, name):
        if name.endswith(".py"):
            return name[:-3]
        return name
