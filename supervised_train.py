import argparse
import threading
import multiprocessing
import gc

from tqdm import tqdm
import mlflow

from mjaigymml.features.feature_analyser_factory import FeatureAnalyzerFactory
from mjaigymml.features.feature_analyser import FeatureAnalyser
from mjaigymml.storage.local_file_mjson_storage import LocalFileMjsonStorage
from mjaigymml.storage.mjson_storage import MjsonStorage
from mjaigymml.config.extract_config import ExtractConfig
from mjaigymml.config.train_config import TrainConfig
from mjaigymml.config.model_config import ModelConfig
from loggers import logger_main as lgs
from mjaigymml.models.model_factory import ModelFactory
from mjaigymml.trainer import Trainer
# import objgraph


TEST_DATASET = multiprocessing.Queue()


def get_test_dataset(test_mjson_storage):
    # テストデータセット作成（初回に作成したものを使いまわす）
    if len(TEST_DATASET) == 0:
        # データ生成
        pass

    return TEST_DATASET


def _run_onegame_analyze(args):
    try:
        mjson, analyser, dataset_queue, train_config = args
        datasets = analyser.analyse_mjson(mjson)
        # extract and sampling.
        datasets = analyser.filter_datasets(datasets, train_config)
        if len(datasets) == 0:
            return
        # calc_feature() updates datasets object
        analyser.calc_feature(datasets)

        dataset_queue.put(datasets)
    except KeyboardInterrupt:
        return
    except Exception as e:
        print(e)
        return


def run_extract_process(
    train_mjson_storage: MjsonStorage,
    analyser: FeatureAnalyser,
    dataset_queue: multiprocessing.Queue,
    train_config: TrainConfig
):
    """
    特徴量抽出を行う
    """

    lgs.info("start extract process")

    cpu_num = multiprocessing.cpu_count()
    # cpu_num = 1  # for debug

    if cpu_num == 1:
        for mjson in train_mjson_storage.get_mjsons():
            arg = (mjson, analyser, dataset_queue, train_config)
            _run_onegame_analyze(arg)
    else:
        one_chunk = []
        with tqdm(total=train_mjson_storage.max_num) as t:
            for mjson in train_mjson_storage.get_mjsons():
                one_chunk.append(
                    (mjson, analyser, dataset_queue, train_config))

                if len(one_chunk) < 128:
                    continue

                with multiprocessing.Pool(processes=cpu_num) as pool:
                    for _ in pool.imap_unordered(_run_onegame_analyze,
                                                 one_chunk):
                        t.update(1)
                    pool.close()
                one_chunk.clear()

                # objgraph.show_growth()


def run(
        train_mjson_dir,
        test_mjson_dir,
        extract_config,
        model_config,
        train_config,
        model_save_dir,
        load_model_file,
):

    # 牌譜読み込み定義
    train_mjson_storage = LocalFileMjsonStorage(
        train_mjson_dir, 400000)  # 10000牌譜ファイル分抽出
    # test_mjson_storage = LocalFileMjsonStorage(
    #     test_mjson_dir, 100)  # 100牌譜ファイル分抽出

    # 牌譜解析の設定
    analyser = FeatureAnalyzerFactory.get_analyzer(
        train_config.model_type,
        extract_config)

    # トレーニングデータセット連携用キュー
    m = multiprocessing.Manager()
    # 1ゲーム分のリストオブジェクトを突っ込むのでcpu数の定数倍用意すればOK
    dataset_queue = m.Queue(
        maxsize=multiprocessing.cpu_count()*2)

    # トレーニングデータセット抽出プロセス
    p = threading.Thread(
        target=run_extract_process,
        args=(train_mjson_storage,
              analyser,
              dataset_queue,
              train_config),
        daemon=True
    )

    # 特徴量消費側定義
    # モデル定義

    model = ModelFactory.get_model(
        train_config,
        model_config,
        analyser.get_reach_dahai_feature_length(),
        analyser.get_pon_chi_kan_feature_length(),
    )
    # 学習の定義

    trainer = Trainer()

    mlflow.set_experiment(train_config.model_type)
    mlflow.start_run()
    mlflow.log_param("train_mjson_dir", train_mjson_dir)
    mlflow.log_param("test_mjson_dir", test_mjson_dir)
    mlflow.log_params(model_config.get_dict())
    mlflow.log_params(train_config.get_dict())
    mlflow.log_param("extract_config", extract_config.get_json())

    p.start()

    # 学習の実行
    trainer.train_loop(
        model,
        dataset_queue,
        p,
        train_config,
        model_config,
        model_save_dir,
        load_model_file
    )

    p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_mjson_dir", type=str,
                        default="/data/mjson/train/")
    parser.add_argument("--test_mjson_dir", type=str,
                        default="/data/mjson/test/201712")
    parser.add_argument("--extract_config", type=str,
                        default="extract_config.yml")
    parser.add_argument("--model_config", type=str,
                        default="model_config.yml")
    parser.add_argument("--train_config", type=str,
                        default="train_config.yml")
    parser.add_argument("--model_save_dir", type=str,
                        default="output/model")
    parser.add_argument("--load_model_file", type=str, default=None)

    arg = parser.parse_args()

    extract_config = ExtractConfig.load(arg.extract_config)
    model_config = ModelConfig.load(arg.model_config)
    train_config = TrainConfig.load(arg.train_config)

    run(
        arg.train_mjson_dir,
        arg.test_mjson_dir,
        extract_config,
        model_config,
        train_config,
        arg.model_save_dir,
        arg.load_model_file,
    )
