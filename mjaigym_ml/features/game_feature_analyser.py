import os
import sys
from pathlib import Path
from typing import Dict, List
import importlib

from mjaigym.mjson import Mjson
from mjaigym.board.board_state import BoardState
from .game_feature_analysis import GameFeatureAnalysis
from mjaigym.board.archive_board import ArchiveBoard

class GameFeatureAnalyser():
    """
    ゲーム解析情報（mjsonクラス）をもとにGameFeatureAnalysisを作成する。
    """

    def __init__(self, feature_extractor_config:Dict):
        self.feature_extractor_config = feature_extractor_config
        self.feature_extractors = []

        # configで指定された特徴量抽出クラスをimportする
        for module_name, target_class_name in self.feature_extractor_config.items():            
            target_module = importlib.import_module(f"mjaigym_ml.features.custom.{module_name}")
            target_class = getattr(target_module, target_class_name)
            self.feature_extractors.append(target_class)
        

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

                # check need to calclate feature
                # if all flag is false, skip calclate.
                if self._need_calclate(state):
                    feature = self.feature_extractors
                
                next_action = None if line_index == len(kyoku.kyoku_mjsons) \
                    else kyoku.kyoku_mjsons[line_index+1]
                
                dahai, reach, pon, chi, kan = \
                    self._possible_action_types(state, next_action)

                # calclate feature for each possible actions
                feature = None                

            line_count += len(kyoku.kyoku_mjsons)
        
        analysis = GameFeatureAnalysis(
            Path(mjson.path), 
            labels, 
            features, 
            self.feature_extractor_config)
        return analysis


    def _need_calclate(state:BoardState):
        if all([len(actions) == 1 for actions in state.possible_actions]):
            return False
        return True

    def _possible_action_types(state:BoardState, next_action:Dict):
        dahai = False
        reach = False
        pon = False
        chi = False
        kan = False
        
        if next_action["type"] == "dahai" and (not state.reachs[next_action["actor"]]):
            dahai = True

        # other check
        for player_actions in state.possible_actions:
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
        



                

        
    
    