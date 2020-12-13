""" feature for furo base class

Raises:
    NotImplementedError: [description]

Returns:
    [type]: [description]
"""
from typing import Dict
from abc import ABCMeta, abstractmethod, abstractclassmethod

import numpy as np

from mjaigym.board import BoardState


class FeaturePonChiKan(metaclass=ABCMeta):
    """
    盤面情報をもとに特徴量を計算するクラス
    """
    @abstractclassmethod
    def get_length(cls) -> int:
        raise NotImplementedError()

    @abstractmethod
    def calc(
            self,
            result: np.array,
            board_state: BoardState,
            player_id: int,
            candidate_furo_action: Dict):
        """
        指定されたプレーヤーidについて特徴量計算を行う
        特徴量は 第一引数 result に記録する
        """
        raise NotImplementedError()

    def reset(self):
        """
        キャッシュなどを以前計算した状態を再利用している場合はキャッシュリセットを行う
        """
        pass
