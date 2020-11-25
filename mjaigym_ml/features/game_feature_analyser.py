import os
import sys
from pathlib import Path
from typing import Dict, List
import importlib

from mjaigym.mjson import Mjson
from mjaigym.board.board_state import BoardState
from .game_feature_analysis import GameFeatureAnalysis
from mjaigym.board.archive_board import ArchiveBoard
from .extractor_config import ExtractorConfig

class GameFeatureAnalyser():
    """
    ゲーム解析情報（mjsonクラス）をもとにGameFeatureAnalysisを作成する。
    """

    def __init__(self, extractor_config:ExtractorConfig):
        self.extractor_config = extractor_config
        self.reach_dahai_extractors = []
        self.pon_chi_kan_extractors = []

        # configで指定された特徴量抽出クラスをimportする
        for module_name, target_class_name in self.extractor_config.on_reach_dahai.items():
            target_module = importlib.import_module(f"mjaigym_ml.features.custom.{module_name}")
            target_class = getattr(target_module, target_class_name)
            self.reach_dahai_extractors.append(target_class)

        for module_name, target_class_name in self.extractor_config.on_pon_chi_kan.items():
            target_module = importlib.import_module(f"mjaigym_ml.features.custom.{module_name}")
            target_class = getattr(target_module, target_class_name)
            self.pon_chi_kan_extractors.append(target_class)


    def analyze_list(self, mjson_list:List[Dict])->GameFeatureAnalysis:
        """
        辞書のリストで与えられたアクションの履歴について特徴量の算出を行う。
        最後の状態について特徴量の算出を行う。
        """
        raise NotImplementedError()

    def analyse_mjson(self, mjson:Mjson)->GameFeatureAnalysis:
        """
        Mjsonオブジェクトで与えられたアクションの履歴について特徴量の算出を行う。
        全アクションについてアクション適用後の特徴量の算出を行う。
        """

        labels = []
        features = []

        dahais = []
        reaches = []
        pons = []
        chis = []
        kans = []
        next_actions = []
        
        board = ArchiveBoard()

        line_count = 0
        for kyoku in mjson.game.kyokus:
            for line_index, action in enumerate(kyoku.kyoku_mjsons):
                board.step(action)
                state = board.get_state()

                # それぞれのモデルの学習対象局面でない行はスキップ
                if not self._need_calclate(state):
                    continue
                
                if line_index+1 >= len(kyoku.kyoku_mjsons):
                    continue
                
                # 次のアクションを取得
                next_action = kyoku.kyoku_mjsons[line_index+1]

                # どのモデルの学習対象かチェック
                dahai, reach, pon, chi, kan = \
                    self._possible_action_types(state, next_action)

                # 特徴量を計算
                feature = None

                # 副露特徴量を計算
                furo_feature = None


            line_count += len(kyoku.kyoku_mjsons)
        
        analysis = GameFeatureAnalysis(
            Path(mjson.path), 
            labels, 
            features, 
            self.extractor_config)
        return analysis


    def _need_calclate(self, state:BoardState):
        if all([len(actions) == 1 for actions in state.possible_actions.values()]):
            return False
        return True

    def _possible_action_types(self, state:BoardState, next_action:Dict):
        dahai = False
        reach = False
        pon = False
        chi = False
        kan = False
        
        if next_action["type"] == "dahai" and (not state.reach[next_action["actor"]]):
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
        



                

        
    
    