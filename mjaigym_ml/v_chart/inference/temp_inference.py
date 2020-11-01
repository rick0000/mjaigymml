import os
import json

from ml.custom_observer import SampleCustomObserver
from mjaigym.board import ArchiveBoard
from mjaigym.reward import KyokuScoreReward


def analyze(mjson_list):
    states = []
    env = SampleCustomObserver(
        board=ArchiveBoard(),
        reward_calclator_cls=KyokuScoreReward)
    env.reset()

    for index, action in enumerate(mjson_list):
        next_state, reward, done, info = env.step(action)

        for player_id in range(4):
            if next_state[player_id].dahai_observation is not None:
                states.append([
                    index,
                    player_id,
                    next_state[player_id].dahai_observation])
    return states


def get_model():
    from ml.model import Head34Value1SlModel
    env = SampleCustomObserver(
        board=ArchiveBoard(),
        reward_calclator_cls=KyokuScoreReward)
    model = Head34Value1SlModel(
        in_channels=env.get_tsumo_observe_channels_num(),
        mid_channels=256,
        blocks_num=50,
        learning_rate=0.0001,
        batch_size=256
    )

    model.load("output/logs/20201017_080649/103522/dahai.pth")
    return model


def load_paifu(fpath):
    if not os.path.isfile(fpath):
        raise Exception("not found")
    try:
        f = open(fpath)
        file_lines = f.readlines()
        f.close()
        jsoned_list = [json.loads(line) for line in file_lines]
        return jsoned_list
    except Exception as e:
        print(e)
        raise Exception("not found")


def main():
    mjson_list = load_paifu("sample.mjson")
    states = analyze(mjson_list)
    for s in states:
        print(s[2].shape)


main()
