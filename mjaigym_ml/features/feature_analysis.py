from pathlib import Path
from typing import Dict, NamedTuple, List

import numpy as np
import pandas as pd


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


class FeatureAnalysis():
    """
    1ゲーム分の牌譜解析情報
    """

    def __init__(
            self,
            fname: Path,
            labels: List[LabelRecord],
            features: Dict[str, np.array]):

        self.fname = fname
        self.labels = pd.DataFrame(data=labels)
        self.features = features

    def get_dahai_records(self):
        pass

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
