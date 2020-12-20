from dataclasses import dataclass
from typing import Dict, NamedTuple

import numpy as np

from mjaigym.board.board_state import BoardState


@dataclass
class LabelRecord():
    filename: str  # mjsonファイル名
    mjson_line_index: int  # 牌譜の何行目まで適用済みの状態か 0-index。
    kyoku_line_index: int  # 局の何行目まで適用済みの状態か 0-index。
    kyoku_line_num: int  # その局の牌譜行数
    kyoku_index: int  # 局の通し番号 0-index
    kyoku: int  # 局の番号 [1,2,3,4]
    bakaze: str  # 場風 ['E','S','W','N']
    honba: int  # 本場
    kyotaku: int  # 供託

    candidate_action_type: str  # 副露行動候補のタイプ
    next_action_type: str  # その後取られたアクションのタイプ

    dahai: bool  # 打牌選択局面の場合1、それ以外0
    reach: bool  # リーチ可能局面の場合1、それ以外0
    chi: bool  # チー可能局面の場合1、それ以外0
    pon: bool  # ポン可能局面の場合1、それ以外0
    kan: bool  # カン可能局面の場合1、それ以外0

    score_diff_0: int  # スコア差分
    score_diff_1: int
    score_diff_2: int
    score_diff_3: int
    initial_score_0: int  # 局開始時スコア
    initial_score_1: int
    initial_score_2: int
    initial_score_3: int
    end_score_0: int  # 局終了時スコア
    end_score_1: int
    end_score_2: int
    end_score_3: int

    candidate_action: Dict  # 副露候補
    next_action: Dict  # その後取られたアクション

    def set_candidate_action(self, candidate_action):
        self.candidate_action = candidate_action

    def set_candidate_action_type(self, candidate_action_type):
        self.candidate_action_type = candidate_action_type


@dataclass
class FeatureRecord():
    """
    board_stateから求めた特徴量
    必要になった際に特徴量の生成を行う。
    データオーグメンテーションもここで対応する。
    Ex) * アクションプレーヤーによる席並び替え
        * 牌種類のローテーション
    """
    common_feature: np.array
    reach_dahai_feature: Dict[int, np.array]  # 席、特徴量の辞書
    reach_dahai_oracle_feature: Dict[int, np.array]  # 席、特徴量の辞書
    pon_chi_kan_feature: Dict[int, np.array]  # 席、特徴量の辞書


class Dataset():
    def __init__(
        self,
        label: LabelRecord,
        board_state: BoardState,
        candidate_action: Dict = None,  # 副露用
    ):
        self.label = label
        self.board_state = board_state
        self.candidate_action = candidate_action
        self.feature = None
        self.is_calclated = False

    def set_feature(self, feature: FeatureRecord):
        self.feature = feature
