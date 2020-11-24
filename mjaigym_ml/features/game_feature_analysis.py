
from typing import Dict

from mjaigym_ml.features.custom.feature import Feature


class GameFeatureAnalysis():
    """
    1ゲーム分の牌譜解析情報
    
    labelsのデータフレームのカラムは以下の通り
    ---
    line: 牌譜の何行目まで適用済みの状態か。0-index。
    action: 現在の状態に対して実際に適用されたアクション。（line+1行のアクション）Dict型。
    kyoku_index: 局の通し番号
    kyoku: 局
    bakaze: 場風
    honba: 本場
    kyotaku: 供託
    
    dahai: 打牌選択局面の場合1、それ以外0
    reach: リーチ可能局面の場合1、それ以外0
    chi: チー可能局面の場合1、それ以外0
    pon: チー可能局面の場合1、それ以外0
    kan: チー可能局面の場合1、それ以外0
    
    score_diff_0: スコア差分
    score_diff_1: 
    score_diff_2: 
    score_diff_3: 
    initial_score_0: 局開始時スコア
    initial_score_1: 
    initial_score_2: 
    initial_score_3:
    end_score_0: 局終了時スコア
    end_score_1: 
    end_score_2: 
    end_score_3: 
    """
        
    def __init__(self, fname, labels, features, feature_extractor_config):
        self.fname = fname
        self.labels = labels
        self.features = features
        self.feature_extractor_config = feature_extractor_config

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



