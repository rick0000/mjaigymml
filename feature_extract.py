import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mjaigym_ml.storage.feature_storage_localfs import FeatureStorageLocalFs
from mjaigym.mjson import Mjson

one_channel_num = 10


def _get_random_data():
    shape = (4, one_channel_num, 34)
    return np.random.randint(0, 2, shape).astype('int8')


randoms = [_get_random_data() for _ in range(100)]


def get_random_data():
    # return np.zeros((4, 10, 34), dtype='int8')
    return randoms[np.random.randint(100)]


def run(mjson_dir, feature_dir, extract_config):
    mjson_paths = Path(mjson_dir).glob("**/*.mjson")
    mjson_paths = list(mjson_paths)

    localfs_path = Path(feature_dir)
    storage = FeatureStorageLocalFs(localfs_path)
    line_num = 500

    start = datetime.datetime.now()
    for mjson_index, mjson_path in enumerate(mjson_paths):
        print(mjson_index, datetime.datetime.now(), mjson_path.name)

        # create labels
        mjson = Mjson.load(mjson_path)
        for kyoku in mjson.game.kyokus:
            for line in kyoku.kyoku_mjsons:

        features = []
        for i in range(line_num):
            line_features = {}
            for f_num in range(25):
                key = f"fature_name{f_num}/{i}"
                line_features[key] = get_random_data()
            features.append(line_features)
            labels.append(LabelRecords)
        storage.save(
            mjson_path.name,
            feature=features,
            label=labels
        )
        if mjson_index == 100:
            break

    print(len(mjson_paths))
    print(datetime.datetime.now() - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjson_dir", type=str)
    parser.add_argument("--feature_dir", type=str,
                        default='./output/extract')
    parser.add_argument("--extract_config", type=str,
                        default='./extract_config.yml')

    args = parser.parse_args()

    run(
        args.mjson_dir,
        args.feature_dir,
        args.extract_config,
    )
