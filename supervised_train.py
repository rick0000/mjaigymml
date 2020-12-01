import argparse
import multiprocessing
from multiprocessing import Pool

from tqdm import tqdm

from mjaigym_ml.features.feature_analyser import FeatureAnalyser
from mjaigym_ml.storage.local_file_mjson_storage import LocalFileMjsonStorage
from mjaigym_ml.storage.feature_storage_localfs \
    import FeatureStorageLocalFs
from mjaigym_ml.features.extract_config import ExtractConfig




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
    

    exit(0)
    # 牌譜読み込み定義
    # 牌譜データ抽出者定義
    # 抽出プロセス実行

    # モデル定義
    # 学習機定義
    # 実行

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--model_type", type=str, default="dahai")
    parser.add_argument("--train_mjson_dir", type=str, default="/data/mjson/train/201701")
    parser.add_argument("--test_mjson_dir", type=str, default="/data/mjson/test/201712")
    parser.add_argument("--extract_config", type=str, default="extract_config.yml")
    parser.add_argument("--model_config", type=str, default="model_config.yml")
    parser.add_argument("--train_config", type=str, default="train_config.yml")
    parser.add_argument("--model_save_dir", type=str, default="output/model")
    parser.add_argument("--model_dir", type=str, default=None)

    arg = parser.parse_args()

    config = ExtractConfig.load(arg.extract_config)
    run(
        arg.model_type, 
        arg.train_mjson_dir, 
        arg.test_mjson_dir,
        arg.extract_config,
        arg.model_config,
        arg.train_config,
        arg.model_save_dir,
        arg.model_dir,
        )
