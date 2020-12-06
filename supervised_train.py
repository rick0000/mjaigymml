import argparse
import threading
from collections import deque
import queue
import multiprocessing

from tqdm import tqdm

from mjaigym_ml.features.feature_analyser_factory import FeatureAnalyzerFactory
from mjaigym_ml.features.feature_analyser import FeatureAnalyser
from mjaigym_ml.storage.local_file_mjson_storage import LocalFileMjsonStorage
from mjaigym_ml.storage.mjson_storage import MjsonStorage
from mjaigym_ml.storage.feature_storage_localfs \
    import FeatureStorageLocalFs
from mjaigym_ml.config.extract_config import ExtractConfig
from mjaigym_ml.config.train_config import TrainConfig
from loggers import logger_main as lgs


TEST_DATASET = multiprocessing.Queue()


def get_test_dataset(test_mjson_storage):
    # テストデータセット作成（初回に作成したものを使いまわす）
    if len(TEST_DATASET) == 0:
        # データ生成
        pass

    return TEST_DATASET


def _run_onegame_analyze(args):
    mjson, analyser, dataset_queue, train_config = args
    datasets = analyser.analyse_mjson(mjson)
    # extract and sampling.
    datasets = analyser.filter_datasets(datasets, train_config)

    # calc_feature() updates datasets object
    analyser.calc_feature(datasets)

    dataset_queue.put(datasets)


def run_extract_process(
    train_mjson_storage: MjsonStorage,
    analyser: FeatureAnalyser,
    dataset_queue: multiprocessing.Queue,
    train_config: TrainConfig
):
    """
    特徴量抽出をマルチプロセスで行う
    """

    lgs.info("start extract process")
    args = [
        (mjson, analyser, dataset_queue, train_config)
        for mjson in train_mjson_storage.get_mjsons()]

    cpu_num = 1
    if cpu_num == 1:
        for arg in args:
            _run_onegame_analyze(arg)
    else:
        with multiprocessing.Pool(processes=cpu_num) as pool:
            # with tqdm(total=len(args)) as t:
            for _ in pool.imap_unordered(_run_onegame_analyze, args):
                pass
                # t.update(1)


def run_train(
    trainer,
    model,
    dataset_queue,
    feature_generate_process
):

    lgs.info("start train")
    while True:
        datasets = deque()

        while True:
            if len(datasets) < 256:
                try:
                    one_mjson_dataset = dataset_queue.get(timeout=1)
                    datasets.extend(one_mjson_dataset)
                except queue.Empty:
                    if not feature_generate_process.is_alive():
                        lgs.info("generate process finished, end train.")
                        return
            else:
                lgs.info("filled queue, start train")
                train_data = list(datasets)
                datasets.clear()
                # trainer.train()


def run(
        model_type,
        train_mjson_dir,
        test_mjson_dir,
        extract_config,
        model_config,
        train_config,
        model_save_dir,
        model_dir,
):

    # 牌譜読み込み定義
    train_mjson_storage = LocalFileMjsonStorage(
        train_mjson_dir, 100)  # 10000牌譜ファイル分抽出
    test_mjson_storage = LocalFileMjsonStorage(
        test_mjson_dir, 100)  # 100牌譜ファイル分抽出

    # 牌譜解析の設定
    analyser = FeatureAnalyzerFactory.get_analyzer(model_type, extract_config)

    # トレーニングデータセット連携用キュー
    m = multiprocessing.Manager()
    dataset_queue = m.Queue()

    # トレーニングデータセット抽出プロセス起動
    p = threading.Thread(
        target=run_extract_process,
        args=(train_mjson_storage,
              analyser,
              dataset_queue,
              train_config),
        daemon=True
    )
    p.start()

    # 特徴量消費側定義
    # モデル定義
    model = None
    # 学習の定義
    trainer = None
    # 学習の実行
    run_train(
        trainer,
        model,
        dataset_queue,
        p)
    p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="dahai")
    parser.add_argument("--train_mjson_dir", type=str,
                        default="/data/mjson/train/201701")
    parser.add_argument("--test_mjson_dir", type=str,
                        default="/data/mjson/test/201712")
    parser.add_argument("--extract_config", type=str,
                        default="extract_config.yml")
    parser.add_argument("--model_config", type=str, default="model_config.yml")
    parser.add_argument("--train_config", type=str, default="train_config.yml")
    parser.add_argument("--model_save_dir", type=str, default="output/model")
    parser.add_argument("--model_dir", type=str, default=None)

    arg = parser.parse_args()

    extract_config = ExtractConfig.load(arg.extract_config)
    model_config = ModelConfig.load(arg.model_config)
    train_config = TrainConfig.load(arg.train_config)

    run(
        arg.model_type,
        arg.train_mjson_dir,
        arg.test_mjson_dir,
        extract_config,
        arg.model_config,
        train_config,
        arg.model_save_dir,
        arg.model_dir,
    )
