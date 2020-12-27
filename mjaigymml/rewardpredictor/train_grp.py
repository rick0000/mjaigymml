import pandas as pd

from mjaigymml.rewardpredictor.global_reward_predictor import GlobalRewardPredictor


def main():
    grp = GlobalRewardPredictor()
    # train = pd.read_pickle("output/dataset/train_grp_dataset.pkl")
    test = pd.read_pickle("output/dataset/test_grp_dataset.pkl")
    print(test)
    grp.train(test)


if __name__ == "__main__":
    main()
