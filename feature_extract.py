import multiprocessing
from multiprocessing import Pool

from tqdm import tqdm

from mjaigym_ml.features.feature_analyser import FeatureAnalyser
from mjaigym_ml.storage.local_file_mjson_storage import LocalFileMjsonStorage
from mjaigym_ml.storage.feature_storage_localfs \
    import FeatureStorageLocalFs
from mjaigym_ml.features.extract_config import ExtractConfig


def analyse_mjson(args):
    mjson, extractor_config, input_dir, output_dir = args
    analyser = FeatureAnalyser(extractor_config)
    feature_storage = FeatureStorageLocalFs(output_dir)
    game_feature_analysis = analyser.analyse_mjson(mjson)
    features, labels = game_feature_analysis.get_records()
    feature_storage.save(mjson.path.name, features, labels)


def run(extractor_config, input_dir, output_dir):
    mjson_storage = LocalFileMjsonStorage(input_dir)

    # save config

    # analyze feature
    all_mjsons = mjson_storage.get_mjsons()

    mjsons = []
    for i, m in enumerate(all_mjsons):
        if i == 1000:
            break
        mjsons.append(m)

    args = [(mjson, extractor_config, input_dir, output_dir)
            for mjson in mjsons]

    # for arg in tqdm(args):
    #     analyse_mjson(*arg)

    cpu_num = multiprocessing.cpu_count()-1
    # cpu_num = 1
    with Pool(processes=cpu_num) as pool:
        with tqdm(total=len(args)) as t:
            for _ in pool.imap_unordered(analyse_mjson, args):
                t.update(1)


if __name__ == "__main__":
    input_dir = "/data/mjson/train/201701"
    output_dir = "output/extract"

    config = ExtractConfig.load("extract_config.yml")
    run(config, input_dir, output_dir)
