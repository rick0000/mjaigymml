""" feature base class

Raises:
    NotImplementedError: [description]

Returns:
    [type]: [description]
"""
from abc import ABCMeta, abstractmethod, abstractclassmethod
from mjaigym.board import BoardState
from mjaigym.board.function.pai import Pai
from typing import List, Tuple
import enum
import numpy as np
import pprint
import os


class Feature(metaclass=ABCMeta):
    """
    盤面情報をもとに特徴量を計算するクラス
    """
    @abstractclassmethod
    def get_length(cls)->int:
        raise NotImplementedError()
        
    @abstractclassmethod
    def calc(cls, result:np.array, board_state:BoardState, player_id:int):
        """
        指定されたプレーヤーidについて特徴量計算を行う
        特徴量は 第一引数 result に記録する
        """
        raise NotImplementedError()

