import os
import gc
from abc import ABCMeta, abstractmethod
import numpy as np
from typing import List

import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from mjaigym.board.function.pai import Pai
from mjaigymml.features.feature_analysis import Dataset, FeatureRecord
from mjaigymml.models.net import BinaryNet, ActorCriticNet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EPS = 10**-9


class Model(metaclass=ABCMeta):
    def __init__(
            self,
            in_channels: int,
            mid_channels: int,
            blocks_num: int,
            learning_rate: float,
            batch_size: int):

        self.in_channels = in_channels
        model = self.build_model(in_channels, mid_channels, blocks_num)
        self.model = model.to(DEVICE)
        self.loss = []
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate)
        self.criterion = self.get_criterion()
        self.softmax = nn.Softmax(dim=1)
        self.batch_size = batch_size
        self.sigmoid = nn.Sigmoid()

    def load(self, path):
        state = torch.load(path, map_location=torch.device(DEVICE))
        self.set_state(state)

    def save(self, path):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)

    def get_state(self):
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

    def set_state(self, state):
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])

    @abstractmethod
    def build_model(
            self,
            in_channels: int,
            mid_channels: int,
            blocks_num: int):
        raise NotImplementedError()

    @abstractmethod
    def get_criterion(self):
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, experiences):
        raise NotImplementedError()

    @abstractmethod
    def update(self, experiences):
        raise NotImplementedError()


class BinaryModel(Model):
    """立直、チー、ポン、カン用モデル
    """

    def __init__(
            self,
            in_channels: int,
            mid_channels: int,
            blocks_num: int,
            learning_rate: float,
            batch_size: int):
        super().__init__(
            in_channels,
            mid_channels,
            blocks_num,
            learning_rate,
            batch_size)

    def build_model(
            self,
            in_channels: int,
            mid_channels: int,
            blocks_num: int):
        return BinaryNet(in_channels, mid_channels, blocks_num)

    def get_criterion(self):
        raise NotImplementedError()

    def policy(self, datasets: List[Dataset]):
        """
        softmaxを適用した行動確率を返す。
        """
        states = np.array([self._calc_feature(d) for d in datasets])

        self.model.eval()
        with torch.no_grad():
            inputs = torch.Tensor(states).float().to(DEVICE)
            policy = self.model(inputs)
            prob = self.softmax(policy)
        return prob.cpu().detach().numpy()

    def update(self, datasets: List[Dataset]):
        """
        ニューラルネットの重みを更新する
        """
        batch_num = len(datasets) // self.batch_size
        if batch_num == 0:
            return {}

        total_loss = 0.0
        correct = 0
        total = 0

        states = np.array([self._calc_feature(d) for d in datasets])
        actions = np.array([self._calc_label(d) for d in datasets])

        # lgs.logger_main.info(
        #   f"start size:{sys.getsizeof(states)//(1024*1024)}MB")
        all_inputs = torch.Tensor(states).float().to(DEVICE)
        all_targets = torch.Tensor(actions).float().to(DEVICE)
        # lgs.logger_main.info(
        #   f"start train {len(experiences)}records to {batch_num} minibatchs")

        result = {}
        all_p_loss = 0
        predicts = []
        # all_v_mse = 0
        for i in range(batch_num):
            inputs = all_inputs[i*self.batch_size:(i+1)*self.batch_size]
            targets = all_targets[i*self.batch_size:(i+1)*self.batch_size]
            self.model.train()
            outputs = self.model(inputs)
            policy_loss = self.criterion(outputs, targets)
            loss = policy_loss

            all_p_loss += policy_loss.detach()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            activated_outputs = self.sigmoid(outputs.data.detach())
            rounded_output = torch.round(activated_outputs)
            correct += rounded_output.eq(targets.data).detach().cpu().sum()

            total += len(inputs)
            total_loss += loss.cpu().detach()
            predicts.append(activated_outputs.cpu().detach().numpy())

        concated_predicts = np.concatenate(predicts, axis=0)
        rounded_concated_predicts = np.round(concated_predicts)
        gc.collect()

        acc = 100.0 * correct / (total + EPS)
        result["train/acc"] = float(acc)
        result["train/loss"] = float(total_loss / batch_num)

        # calc pr-auc
        # pr-auc is robust for imbalanced data
        average_precision = average_precision_score(
            actions, concated_predicts)

        result["train/average_precision"] = average_precision

        result["train/1"] = (actions == 1).sum()
        result["train/0"] = (actions == 0).sum()
        result["train/t1_p1"] = ((actions == 1) &
                                 (actions == rounded_concated_predicts)).sum()
        result["train/t0_p0"] = ((actions == 0) &
                                 (actions == rounded_concated_predicts)).sum()
        result["train/t1_p0"] = ((actions == 1) &
                                 (actions != rounded_concated_predicts)).sum()
        result["train/t0_p1"] = ((actions == 0) &
                                 (actions != rounded_concated_predicts)).sum()

        return result

    def evaluate(self, datasets: List[Dataset]):
        batch_num = len(datasets) // self.batch_size
        if batch_num == 0:
            return {}

        total_loss = 0.0
        correct = 0
        total = 0

        states = np.array([self._calc_feature(d) for d in datasets])
        actions = np.array([self._calc_label(d) for d in datasets])

        # lgs.logger_main.info(
        #   f"start size:{sys.getsizeof(states)//(1024*1024)}MB")
        all_inputs = torch.Tensor(states).float().to(DEVICE)
        all_targets = torch.Tensor(actions).float().to(DEVICE)
        # lgs.logger_main.info(
        #   f"start train {len(experiences)}records to {batch_num} minibatchs")

        result = {}
        all_p_loss = 0

        # all_v_mse = 0
        for i in range(batch_num):
            inputs = all_inputs[i*self.batch_size:(i+1)*self.batch_size]
            targets = all_targets[i*self.batch_size:(i+1)*self.batch_size]
            self.model.eval()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            all_p_loss += loss.detach()

            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum().detach()
            total += len(inputs)
            total_loss += loss.cpu().detach()

        gc.collect()

        acc = 100.0 * correct / (total + EPS)
        result["test/dahai_acc"] = float(acc)
        result["test/loss"] = float(total_loss / batch_num)
        result["test/dahai_loss"] = all_p_loss / batch_num

        return result

    def _calc_feature(self, dataset: Dataset):
        raise NotImplementedError()

    def _calc_label(self, dataset: Dataset):
        raise NotImplementedError()


class ReachModel(BinaryModel):
    POS_NEG_RATE = torch.Tensor([2.0]).float().to(DEVICE)

    def get_criterion(self):
        # for imbalanced data
        return nn.BCEWithLogitsLoss(pos_weight=ReachModel.POS_NEG_RATE)

    def _calc_feature(self, dataset: Dataset):
        if dataset.label.next_action_type is not None:
            actor = dataset.label.next_action['actor']
        elif dataset.board_state.previous_action["type"] in ["tsumo", "reach"]:
            actor = dataset.board_state.previous_action['actor']
        else:
            import pdb
            pdb.set_trace()
            raise Exception("not implemented")
        shimocha = (actor + 1) % 4
        toimen = (actor + 2) % 4
        kamicha = (actor + 3) % 4

        oracle_zeros = np.zeros_like(
            dataset.feature.reach_dahai_oracle_feature[actor])
        concated = np.concatenate([
            dataset.feature.common_feature,
            dataset.feature.reach_dahai_feature[actor],
            dataset.feature.reach_dahai_feature[shimocha],
            dataset.feature.reach_dahai_feature[toimen],
            dataset.feature.reach_dahai_feature[kamicha],
            dataset.feature.reach_dahai_oracle_feature[actor],
            oracle_zeros,  # shimocha
            oracle_zeros,  # toimen
            oracle_zeros,  # kamicha
        ], axis=0)
        return concated[:, :, np.newaxis]

    def _calc_label(self, dataset: Dataset):
        do_reach = dataset.label.next_action['type'] == "reach"
        return do_reach


class PonModel(BinaryModel):
    POS_NEG_RATE = torch.Tensor([4.0]).float().to(DEVICE)

    def get_criterion(self):
        # for imbalanced data
        return nn.BCEWithLogitsLoss(pos_weight=PonModel.POS_NEG_RATE)

    def _calc_feature(self, dataset: Dataset):

        actor = dataset.candidate_action['actor']
        shimocha = (actor + 1) % 4
        toimen = (actor + 2) % 4
        kamicha = (actor + 3) % 4

        oracle_zeros = np.zeros_like(
            dataset.feature.reach_dahai_oracle_feature[actor])

        concated = np.concatenate([
            dataset.feature.common_feature,
            dataset.feature.reach_dahai_feature[actor],
            dataset.feature.reach_dahai_feature[shimocha],
            dataset.feature.reach_dahai_feature[toimen],
            dataset.feature.reach_dahai_feature[kamicha],
            dataset.feature.reach_dahai_oracle_feature[actor],
            oracle_zeros,  # shimocha
            oracle_zeros,  # toimen
            oracle_zeros,  # kamicha
            dataset.feature.pon_chi_kan_feature,
        ], axis=0)
        return concated[:, :, np.newaxis]

    def _calc_label(self, dataset: Dataset):
        do_pon = dataset.label.next_action == dataset.label.candidate_action
        assert dataset.label.candidate_action["type"] == "pon"
        return do_pon


class ChiModel(BinaryModel):
    POS_NEG_RATE = torch.Tensor([10.0]).float().to(DEVICE)

    def get_criterion(self):
        # for imbalanced data
        return nn.BCEWithLogitsLoss(pos_weight=ChiModel.POS_NEG_RATE)

    def _calc_feature(self, dataset: Dataset):

        actor = dataset.candidate_action['actor']
        shimocha = (actor + 1) % 4
        toimen = (actor + 2) % 4
        kamicha = (actor + 3) % 4

        shimocha = (actor + 1) % 4
        toimen = (actor + 2) % 4
        kamicha = (actor + 3) % 4

        oracle_zeros = np.zeros_like(
            dataset.feature.reach_dahai_oracle_feature[actor])
        concated = np.concatenate([
            dataset.feature.common_feature,
            dataset.feature.reach_dahai_feature[actor],
            dataset.feature.reach_dahai_feature[shimocha],
            dataset.feature.reach_dahai_feature[toimen],
            dataset.feature.reach_dahai_feature[kamicha],
            dataset.feature.reach_dahai_oracle_feature[actor],
            oracle_zeros,  # shimocha
            oracle_zeros,  # toimen
            oracle_zeros,  # kamicha
            dataset.feature.pon_chi_kan_feature,
        ], axis=0)
        return concated[:, :, np.newaxis]

    def _calc_label(self, dataset: Dataset):
        do_chi = dataset.label.next_action == dataset.label.candidate_action
        try:
            assert dataset.label.candidate_action["type"] == "chi"
        except:
            import pdb
            pdb.set_trace()
        return do_chi


class KanModel(BinaryModel):
    POS_NEG_RATE = torch.Tensor([10.0]).float().to(DEVICE)

    def get_criterion(self):
        # for imbalanced data
        return nn.BCEWithLogitsLoss(pos_weight=KanModel.POS_NEG_RATE)

    def _calc_feature(self, dataset: Dataset):
        actor = dataset.candidate_action['actor']
        shimocha = (actor + 1) % 4
        toimen = (actor + 2) % 4
        kamicha = (actor + 3) % 4

        oracle_zeros = np.zeros_like(
            dataset.feature.reach_dahai_oracle_feature[actor])
        concated = np.concatenate([
            dataset.feature.common_feature,
            dataset.feature.reach_dahai_feature[actor],
            dataset.feature.reach_dahai_feature[shimocha],
            dataset.feature.reach_dahai_feature[toimen],
            dataset.feature.reach_dahai_feature[kamicha],
            dataset.feature.reach_dahai_oracle_feature[actor],
            oracle_zeros,  # shimocha
            oracle_zeros,  # toimen
            oracle_zeros,  # kamicha
            dataset.feature.pon_chi_kan_feature,
        ], axis=0)
        return concated[:, :, np.newaxis]

    def _calc_label(self, dataset: Dataset):
        do_kan = dataset.label.next_action == dataset.label.candidate_action
        try:
            assert dataset.label.candidate_action["type"] in [
                "ankan", "kakan", "daiminkan"]
        except:
            import pdb
            pdb.set_trace()
        return do_kan


class DahaiModel(Model):
    """ActorCritic教師あり打牌用モデル
    """

    def __init__(
            self,
            in_channels: int,
            mid_channels: int,
            blocks_num: int,
            learning_rate: float,
            batch_size: int,
            oracle_rate: float = 0.0,
    ):
        super().__init__(
            in_channels,
            mid_channels,
            blocks_num,
            learning_rate,
            batch_size)

        self.celoss = nn.CrossEntropyLoss()
        self.mseloss = nn.MSELoss()
        self.oracle_rate = oracle_rate
        self.no_oracle = oracle_rate == 0.0

    def build_model(
            self,
            in_channels: int,
            mid_channels: int,
            blocks_num: int):
        return ActorCriticNet(in_channels, mid_channels, blocks_num)

    def get_criterion(self):
        def sl_criterion_func(outputs, targets, v_outputs, v_targets):
            value_loss = self.mseloss(v_outputs, v_targets)
            policy_loss = self.celoss(outputs, targets)

            return policy_loss, value_loss

        return sl_criterion_func

    def predict(self, datasets: List[Dataset]):
        """
        softmaxを適用した行動確率と現在の価値の予測値を返す。中間層の値を含める。
        """
        states = np.array([self._calc_feature(d) for d in datasets])

        self.model.eval()
        with torch.no_grad():
            inputs = torch.Tensor(states).float().to(DEVICE)
            p, v, v_mid, p_mid = self.model.forward_with_mid(inputs)
            prob = self.softmax(p)
        return prob.cpu().detach().numpy(), \
            v.cpu().detach().numpy(), \
            v_mid.cpu().detach().numpy(), \
            p_mid.cpu().detach().numpy()

    def policy(self, datasets: List[Dataset]):
        """
        softmaxを適用した行動確率を返す。
        """
        states = np.array([self._calc_feature(d) for d in datasets])

        self.model.eval()
        with torch.no_grad():
            inputs = torch.Tensor(states).float().to(DEVICE)
            policy = self.model(inputs)
            prob = self.softmax(policy)
        return prob.cpu().detach().numpy()

    def update(self, datasets: List[Dataset]):
        """
        ニューラルネットの重みを更新する
        """
        batch_num = len(datasets) // self.batch_size
        if batch_num == 0:
            return {}

        total_loss = 0.0
        correct = 0
        total = 0

        states = np.array([self._calc_feature(d) for d in datasets])
        actions = np.array([self._calc_label(d) for d in datasets])
        rewards = np.array([self._calc_reward(d) for d in datasets])

        # lgs.logger_main.info(
        #   f"start size:{sys.getsizeof(states)//(1024*1024)}MB")
        all_inputs = torch.Tensor(states).float().to(DEVICE)
        all_targets = torch.Tensor(actions).long().to(DEVICE)
        all_v_targets = torch.Tensor(rewards).float().to(DEVICE)
        # lgs.logger_main.info(
        #   f"start train {len(experiences)}records to {batch_num} minibatchs")

        result = {}
        all_p_loss = 0
        all_v_loss = 0
        # all_v_mse = 0
        for i in range(batch_num):
            inputs = all_inputs[i*self.batch_size:(i+1)*self.batch_size]
            targets = all_targets[i*self.batch_size:(i+1)*self.batch_size]
            v_targets = all_v_targets[i*self.batch_size:(i+1)*self.batch_size]
            self.model.train()
            outputs, v_outputs = self.model(inputs)
            policy_loss, value_loss = self.criterion(
                outputs, targets, v_outputs, v_targets)
            loss = policy_loss + value_loss

            all_p_loss += policy_loss.detach()
            all_v_loss += value_loss.detach()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum().detach()
            total += len(inputs)
            total_loss += loss.cpu().detach()

        gc.collect()

        acc = 100.0 * correct / (total + EPS)
        result["train/dahai_acc"] = float(acc)

        result["train/loss"] = float(total_loss / batch_num)
        result["train/dahai_loss"] = float(all_p_loss / batch_num)
        result["train/value_loss"] = float(all_v_loss / batch_num)
        result["train/rewardvar"] = np.var(rewards)
        var_minus_loss = (float(np.var(rewards)) - result["train/value_loss"])
        result["train/RewardvarValuelossDiffRate"] = \
            var_minus_loss / result["train/rewardvar"]

        return result

    def evaluate(self, datasets: List[Dataset]):
        batch_num = len(datasets) // self.batch_size
        if batch_num == 0:
            return {}

        total_loss = 0.0
        correct = 0
        total = 0

        states = np.array([self._calc_feature(d) for d in datasets])
        actions = np.array([self._calc_label(d) for d in datasets])
        rewards = np.array([self._calc_reward(d) for d in datasets])

        # lgs.logger_main.info(
        #   f"start size:{sys.getsizeof(states)//(1024*1024)}MB")
        all_inputs = torch.Tensor(states).float().to(DEVICE)
        all_targets = torch.Tensor(actions).long().to(DEVICE)
        all_v_targets = torch.Tensor(rewards).float().to(DEVICE)
        # lgs.logger_main.info(
        #   f"start train {len(experiences)}records to {batch_num} minibatchs")

        result = {}
        all_p_loss = 0
        all_v_loss = 0
        # all_v_mse = 0
        for i in range(batch_num):
            inputs = all_inputs[i*self.batch_size:(i+1)*self.batch_size]
            targets = all_targets[i*self.batch_size:(i+1)*self.batch_size]
            v_targets = all_v_targets[i*self.batch_size:(i+1)*self.batch_size]
            self.model.eval()
            outputs, v_outputs = self.model(inputs)
            policy_loss, value_loss = self.criterion(
                outputs, targets, v_outputs, v_targets)
            loss = policy_loss + value_loss
            # loss = value_loss

            all_p_loss += policy_loss.detach()
            all_v_loss += value_loss.detach()

            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum().detach()
            total += len(inputs)
            total_loss += loss.cpu().detach()

        gc.collect()

        acc = 100.0 * correct / (total + EPS)
        result["test/dahai_acc"] = float(acc)

        result["test/loss"] = float(total_loss / batch_num)
        result["test/dahai_loss"] = float(all_p_loss / batch_num)
        result["test/value_loss"] = float(all_v_loss / batch_num)
        result["test/rewardvar"] = np.var(rewards)
        result["test/RewardvarValuelossDiffRate"] = \
            (float(np.var(rewards)) -
             result["test/value_loss"]) / result["test/rewardvar"]

        return result

    def _calc_feature(self, dataset: Dataset):
        if dataset.label.next_action_type is not None:
            actor = dataset.label.next_action['actor']
        elif dataset.board_state.previous_action["type"] in ["tsumo", "reach"]:
            actor = dataset.board_state.previous_action['actor']
        else:
            import pdb
            pdb.set_trace()
            raise Exception("not implemented")
        shimocha = (actor + 1) % 4
        toimen = (actor + 2) % 4
        kamicha = (actor + 3) % 4

        if self.no_oracle or self.oracle_rate <= np.random.rand():
            oracle_zeros = np.zeros_like(
                dataset.feature.reach_dahai_oracle_feature[actor])
            concated = np.concatenate([
                dataset.feature.common_feature,
                dataset.feature.reach_dahai_feature[actor],
                dataset.feature.reach_dahai_feature[shimocha],
                dataset.feature.reach_dahai_feature[toimen],
                dataset.feature.reach_dahai_feature[kamicha],
                dataset.feature.reach_dahai_oracle_feature[actor],
                oracle_zeros,  # shimocha
                oracle_zeros,  # toimen
                oracle_zeros,  # kamicha
            ], axis=0)
        else:
            # print("use oracle feature")
            oracle_zeros = np.zeros_like(
                dataset.feature.reach_dahai_oracle_feature[actor])
            concated = np.concatenate([
                dataset.feature.common_feature,
                dataset.feature.reach_dahai_feature[actor],
                dataset.feature.reach_dahai_feature[shimocha],
                dataset.feature.reach_dahai_feature[toimen],
                dataset.feature.reach_dahai_feature[kamicha],
                dataset.feature.reach_dahai_oracle_feature[actor],
                # shimocha
                dataset.feature.reach_dahai_oracle_feature[shimocha],
                dataset.feature.reach_dahai_oracle_feature[toimen],  # toimen
                dataset.feature.reach_dahai_oracle_feature[kamicha],  # kamicha
            ], axis=0)

        return concated[:, :, np.newaxis]

    def _calc_label(self, dataset: Dataset):
        return Pai.str_to_id(dataset.label.next_action['pai'])

    def _calc_reward(self, dataset: Dataset):
        """
        NOTE:1000点を1としている。

        NOTE:以下を使えば割引報酬にすることが可能

        dataset.label.kyoku_line_index
            局の何行目まで適用済みの状態か 0-index。

        dataset.label.kyoku_line_num
            その局の牌譜行数
        """

        actor = dataset.label.next_action['actor']
        diffs = [
            dataset.label.score_diff_0,
            dataset.label.score_diff_1,
            dataset.label.score_diff_2,
            dataset.label.score_diff_3,
        ]

        # 1000点を1とする
        REWARD_BASE = 1000.0
        diffs = [d / REWARD_BASE for d in diffs]

        # 割引
        rate = 0.995 ** (dataset.label.kyoku_line_num -
                         dataset.label.kyoku_line_index)
        diffs = [d * rate for d in diffs]

        pre_actor = diffs[:actor]
        post_actor = diffs[actor:]
        return post_actor + pre_actor


if __name__ == "__main__":
    in_channels = 10
    mid_channels = 128
    blocks_num = 10
    learning_rate = 0.01
    batch_size = 128

    model = DahaiModel(
        in_channels,
        mid_channels,
        blocks_num,
        learning_rate,
        batch_size
    )

    exps = []
    for _ in range(256):
        state = np.random.randint(0, 2, size=(in_channels, 34, 1))
        action = np.random.randint(34)
        reward = np.random.randint(-100, 100, (4,))
        exps.append([state, action, reward])

    model.update(exps)
