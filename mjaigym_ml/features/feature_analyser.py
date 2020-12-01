from pathlib import Path
from typing import Dict, List
import importlib

import numpy as np

from mjaigym.mjson import Mjson
from mjaigym.board.board_state import BoardState
from mjaigym_ml.features.feature_analysis import FeatureAnalysis
from mjaigym.board.archive_board import ArchiveBoard
from .extract_config import ExtractConfig
from mjaigym_ml.features.custom.feature_common import FeatureCommon
from mjaigym_ml.features.custom.feature_reach_dahai import FeatureReachDahai
from mjaigym_ml.features.custom.feature_pon_chi_kan import FeaturePonChiKan
from mjaigym_ml.features.feature_analysis import LabelRecord


class FeatureAnalyser():
    """
    ゲーム解析情報（mjsonクラス）をもとにGameFeatureAnalysisを作成する。
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

    def analyze_list(self, mjson_list: List[Dict]) -> FeatureAnalysis:
        """
        辞書のリストで与えられたアクションの履歴について特徴量の算出を行う。
        全アクションではなく最後の状態についてのみ特徴量の算出を行う。
        """
        raise NotImplementedError()

    def analyse_mjson(self, mjson: Mjson) -> FeatureAnalysis:
        """
        Mjsonオブジェクトで与えられたアクションの履歴について特徴量の算出を行う。
        全アクションについてアクション適用後の特徴量の算出を行う。
        """

        labels = []
        features = {}

        board = ArchiveBoard()

        line_count = 0
        for kyoku_index, kyoku in enumerate(mjson.game.kyokus):
            for kyoku_line_index, action in enumerate(kyoku.kyoku_mjsons):
                board.step(action)
                board_state = board.get_state()

                # ファイル上での行数通し番号
                mjson_line_index = line_count + kyoku_line_index
                # それぞれのモデルの学習対象局面でない行はスキップ
                if not self._need_calclate(board_state):
                    continue

                if kyoku_line_index+1 >= len(kyoku.kyoku_mjsons):
                    continue

                # 次のアクションを取得
                next_action = kyoku.kyoku_mjsons[kyoku_line_index+1]

                # どのモデルの学習対象かチェック
                dahai, reach, pon, chi, kan = self._possible_action_types(
                    board_state, next_action)

                if dahai:
                    # dahaiの教師データを作成
                    record = LabelRecord(
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
                    )

                    labels.append(record)

                    # 特徴量を計算
                    common_features = np.zeros(
                        (1, self.common_length, 34))
                    start_index = 0
                    for f in self.common_extractors:
                        feature_name = f.__class__.__name__
                        target_array = common_features[
                            0,
                            start_index:start_index+f.get_length(),
                            :
                        ]
                        f.calc(target_array, board_state)
                        feature_key = self._get_feature_key(
                            mjson_line_index, feature_name, 0)
                        features[feature_key] = target_array

                

                # 副露特徴量を計算
                reach_dahai_features = np.zeros(
                    (4, self.reach_dahai_length, 34))
                start_index = 0
                for f in self.reach_dahai_extractors:
                    feature_name = f.__class__.__name__
                    for player_id in range(4):
                        target_array = reach_dahai_features[
                            player_id,
                            start_index:start_index+f.get_length(),
                            :
                        ]
                        f.calc(target_array, board_state, player_id)
                        feature_key = self._get_feature_key(
                            mjson_line_index, feature_name, player_id)
                        features[feature_key] = target_array
                """

                if pon or chi or kan:
                    # pon chi kan feature
                    for player_id in range(4):
                        possible_actions = board_state.possible_actions[player_id]
                        for possible_action in possible_actions:
                            pon_chi_kan_features = np.zeros(
                                (4, self.pon_chi_kan_length, 34))
                            start_index = 0
                            for f in self.pon_chi_kan_extractors:
                                feature_name = f.__class__.__name__
                                target_array = pon_chi_kan_features[
                                    player_id,
                                    start_index:start_index+f.get_length(),
                                    :
                                ]
                                f.calc(target_array, board_state, player_id)
                                feature_key = self._get_feature_key(
                                    mjson_line_index, feature_name, player_id)
                                features[feature_key] = target_array
                """
            line_count += len(kyoku.kyoku_mjsons)

        # 作成した1ゲーム分のデータをまとめる
        analysis = FeatureAnalysis(
            Path(mjson.path),
            labels,
            features,
        )
        return analysis

    def _get_feature_key(self, mjson_line_index, feature_name, player_id=None, possible_action_id=None):
        if player_id is None:
            return f"{mjson_line_index}/{feature_name}"
        elif possible_action_id is None:
            return f"{mjson_line_index}/{feature_name}/{player_id}"
        else:
            return f"{mjson_line_index}/{feature_name}/{player_id}/{possible_action_id}"

    def _need_calclate(self, state: BoardState):
        # 選択可能なアクションが複数ない場合は教師データにならない。
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

        if next_action["type"] == "dahai" and \
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
            f"mjaigym_ml.features.custom.{dir_name}.{target_module_name}")
        target_class = getattr(target_module, target_class_name)
        return target_class

    def _filter_dotpy(self, name):
        if name.endswith(".py"):
            return name[:-3]
        return name
