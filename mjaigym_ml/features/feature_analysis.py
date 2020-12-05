from pathlib import Path
from typing import Dict, NamedTuple, List
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mjaigym.board.board_state import BoardState


class LabelRecord(NamedTuple):
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
    next_and_candidate_is_same: bool  # 副露行動候補が次のアクションで採用されたか
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


@dataclass
class FeatureRecord():
    """
    board_stateから求めた特徴量
    計算量削減のため必要になった際に特徴量の生成を行う。
    データオーグメンテーションもここで対応する。
    Ex) * アクションプレーヤーによる席並び替え
        * 牌種類のローテーション
    """
    common_feature: np.array
    reach_dahai_feature: Dict[int, np.array]  # 席、特徴量の辞書
    pon_chi_kan_feature: Dict[int, np.array]  # 席、特徴量の辞書

    def __init__(
            self,
            board_state: BoardState,
            candidate_action: Dict = None,
    ):
        self.board_state = board_state
        self.candidate_action = candidate_action

    def get_feature(
            self,
            common_feature_calc_function,
            reach_dahai_feature_calc_function,
            pon_chi_kan_feature_calc_function,
    ):
        if self.common_feature is None:
            self.common_feature = common_feature_calc_function(
                self.board_state)

        if self.reach_dahai_feature is None:
            self.reach_dahai_feature = reach_dahai_feature_calc_function(
                self.board_state)

        if self.pon_chi_kan_feature is None:
            self.pon_chi_kan_feature = pon_chi_kan_feature_calc_function(
                self.board_state)

        # concat, rotate feature
        concated = np.concatenate(
            [self.common_feature]
            + [self.reach_dahai_feature.values()]
            + [self.pon_chi_kan_feature.values()], axis=0)
        return concated


class Datasets():
    """
    1ゲーム分の牌譜解析情報
    """

    def __init__(
        self,
        fname: Path,
    ):
        self.fname = fname
        self.labels = []
        self.features = []

    def append(self, label: LabelRecord, feature: FeatureRecord):
        self.labels.append(label)
        self.features.append(feature)

    def get_dahai_records(self):
        # 打牌のみ抽出
        result_index = [i for (i, l) in enumerate(self.labels) if l.dahai]

        # 変換
        result_labels = [self.labels[i] for i in range(result_index)]
        result_features = [self.features[i] for i in range(result_index)]

        return pd.DataFrame(data=result_labels), \
            [f.get_feature() for f in result_features]

    def get_reach_records(self):
        pass

    def get_pon_records(self):
        pass

    def get_chi_records(self):
        pass

    def get_kan_records(self):
        pass

    def get_records(self):
        """
        全データを返す
        """
        return self.features, self.labels

    def get_rotate_pai_feature(self):
        """
        [m,p,s,z]の並び順についてm,p,sの入れ替えを行い、
        全6パターン（3*2*1）を生成する。
        """
        pass

    def get_roatete_position_feature(self):
        """
        下家、対面、上家の並び順について入れ替えを行い
        全6パターンを生成する。
        """
        pass
