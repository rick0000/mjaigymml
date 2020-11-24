import time
import copy
import math
import random
import datetime
from collections import deque
from pathlib import Path
import multiprocessing
from multiprocessing import Queue
from multiprocessing import Pool, Process, set_start_method, Queue

import tqdm
import numpy as np
from dataclasses import dataclass

from mjaigym.board.function.pai import Pai
from mjaigym.board.mj_move import MjMove
from mjaigym.mjson import Mjson
from ml.framework import Experience, MjObserver
import mjaigym.loggers as lgs
from mjaigym.board.archive_board import ArchiveBoard


def _analyze_one_game(mjson_path):
    env = ArchiveBoard()
    mjson = Mjson.load(mjson_path)
    for kyoku in mjson.game.kyokus:
        env.reset()
        states = []
        actions = []
        rewards = []
        board_states = []

        dahais = deque()
        chis = deque()
        pons = deque()
        kans = deque()
        reaches = deque()
        
        for action in kyoku.kyoku_mjsons:
            current_state = env.get_state()
            
            # print(current_state.possible_actions)
            items = [(action["type"], 0) for actions in current_state.possible_actions[0]]\
                + [(action["type"], 1) for actions in current_state.possible_actions[1]]\
                + [(action["type"], 2) for actions in current_state.possible_actions[2]]\
                + [(action["type"], 3) for actions in current_state.possible_actions[3]]
            items = set(items)
            
            # print(items)
            env.step(action)


            
            
            
        if len(rewards) == 0:
            continue


if __name__ == "__main__":
    print(datetime.datetime.now(), "start analyze")
    for i in tqdm.tqdm(range(100)):
        _analyze_one_game("sample.mjson")
    print(datetime.datetime.now(), "end analyze")

