import multiprocessing
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm
from mjaigym.mjson import Mjson
from mjaigymml.rewardpredictor.grp_dataset import GrpDataset


def generate_df_from_mjson(mjson: Mjson) -> pd.DataFrame:
    datas = []
    kyotaku = 0
    is_tonnan = False
    chicha = None
    for kyoku in mjson.game.kyokus:
        kyoku_start_line = kyoku.kyoku_mjsons[0]
        kyoku_id = kyoku_start_line["kyoku"]
        bakaze = kyoku_start_line["bakaze"]
        honba = kyoku_start_line["honba"]
        oya = kyoku_start_line["oya"]
        if chicha is None:
            chicha = oya
        reach_accepted_lines = [
            m for m in kyoku.kyoku_mjsons if m["type"] == "reach_accepted"]
        kyotaku += len(reach_accepted_lines)

        # calc dataset
        end_scores = kyoku.kyoku_mjsons[-2]["scores"]
        diff = kyoku.kyoku_mjsons[-2]["deltas"]
        before_scores = [s - d for (s, d) in zip(end_scores, diff)]

        record = GrpDataset(
            before_scores=before_scores,
            end_scores=end_scores,
            label_scores=mjson.game.game_result_score,
            kyoku=kyoku_id,
            bakaze=bakaze,
            honba=honba,
            chicha=chicha,
            oya=oya,
            kyotaku=kyotaku
        )
        datas.append(record)

        # after generate dataset processing
        if kyoku.kyoku_mjsons[-2]["type"] == "hora":
            kyotaku = 0

        if bakaze == "S" and kyoku_id == 4:
            is_tonnan = True

    if len(datas) == 0:
        return pd.DataFrame()

    feature_df = pd.DataFrame([d.feature for d in datas])
    feature_df.loc[:, "is_tonnnan"] = is_tonnan
    label_df = pd.DataFrame([d.labels for d in datas])

    datas_df = pd.concat([feature_df, label_df], axis=1)

    return datas_df


def _run_onegame_analyze(mjson_path):
    mjson = (Mjson.load(mjson_path))
    df = generate_df_from_mjson(mjson)
    return df


def main(mjson_dir):
    mjson_paths = list(Path(mjson_dir).glob("**/*.mjson"))
    use_mjson_paths = mjson_paths

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        imap = pool.imap(_run_onegame_analyze, use_mjson_paths)
        dfs = list(tqdm(imap, total=len(use_mjson_paths)))

    df = pd.concat(dfs, axis=0)
    return df


if __name__ == "__main__":
    train = main("/data/mjson/train/")
    train_output = "./output/dataset/train_grp_dataset"
    Path(train_output).parent.mkdir(parents=True, exist_ok=True)
    train.to_pickle(f"{train_output}.pkl")
    train.to_csv(f"{train_output}.csv", index=False)

    test = main("/data/mjson/test/201712/")
    test_output = "./output/dataset/test_grp_dataset"
    Path(test_output).parent.mkdir(parents=True, exist_ok=True)
    test.to_pickle(f"{test_output}.pkl")
    test.to_csv(f"{test_output}.csv", index=False)
