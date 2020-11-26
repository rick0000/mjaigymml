from pathlib import Path
import datetime

import numpy as np
import pandas as pd

# from mjaigym_ml.storage.feature_storage_hdf5 import FeatureStorageHDF5
from mjaigym_ml.storage.feature_storage_localfs import FeatureStorageLocalFs


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

    # fatureの保存先としてhdf5ファイルを使用
    # hdf5_path = Path("output/compress_test/supervised_dataset")
    # hdf5_path = Path("output/supervised_dataset")
    # hdf5_path.parent.mkdir(exist_ok=True, parents=True)

    # storage = FeatureStorageHDF5(hdf5_path)

    localfs_path = Path("output/localfs/supervised_dataset")
    storage = FeatureStorageLocalFs(localfs_path)
    line_num = 500

    start = datetime.datetime.now()
    for mjson_index, mjson_path in enumerate(mjson_paths):
        print(datetime.datetime.now(), mjson_index, mjson_path.name)

        features = {}
        for i in range(line_num):
            for f_num in range(25):
                key = f"fature_name{f_num}/{i}"
                features[key] = get_random_data()

        labels = pd.DataFrame(
            np.random.randint(0, 2, (line_num, 10), dtype='int8')
        )

        storage.save(
            mjson_path.name,
            feature=features,
            label=labels
        )
        if mjson_index == 100:
            break

    print(len(mjson_paths))
    print(start.start.now() - start)


if __name__ == "__main__":
    run(
        "/data/mjson/train/201701",
        "./output/feature",
        {}
    )
