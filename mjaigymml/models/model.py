import os
import gc
from abc import ABCMeta, abstractmethod
import numpy as np
from typing import List

import torch
import torch.nn as nn

from mjaigym.board.function.pai import Pai
from mjaigymml.features.feature_analysis import Dataset, FeatureRecord
from mjaigymml.models.net import Head2Net, ActorCriticNet

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


class Head2Model(Model):
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
        return Head2Net(in_channels, mid_channels, blocks_num)

    def get_criterion(self):
        return nn.CrossEntropyLoss()

    def policy(self, states):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.Tensor(states).float().to(DEVICE)
            policy = self.model(inputs)
            prob = self.softmax(policy)
        return prob.cpu().detach().numpy()

    def estimate(self, states):
        raise NotImplementedError()

    def evaluate(self, experiences):
        batch_num = len(experiences) // self.batch_size
        if batch_num == 0:
            return 0, 0

        total_loss = 0.0
        correct = 0
        total = 0
        for i in range(batch_num):
            target_experiences = \
                experiences[i * self.batch_size:(i+1)*self.batch_size]
            states = [e[0] for e in target_experiences]
            actions = [e[1] for e in target_experiences]

            self.model.eval()
            with torch.no_grad():
                inputs = torch.Tensor(states).float().to(DEVICE)
                targets = torch.Tensor(actions).long().to(DEVICE)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets.data).cpu().sum().detach()
                total += len(states)
                total_loss += loss.cpu().detach()

            del states, actions, target_experiences, inputs, targets
        gc.collect()
        acc = 100.0 * correct / (total + EPS)
        return float(total_loss / batch_num), float(acc)

    def update(self, experiences):
        batch_num = len(experiences) // self.batch_size
        if batch_num == 0:
            return 0, 0

        total_loss = 0.0
        correct = 0
        total = 0
        for i in range(batch_num):
            target_experiences = \
                experiences[i * self.batch_size:(i+1)*self.batch_size]
            states = [e[0] for e in target_experiences]
            actions = [e[1] for e in target_experiences]

            self.model.train()
            inputs = torch.Tensor(states).float().to(DEVICE)
            targets = torch.Tensor(actions).long().to(DEVICE)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum().detach()
            total += len(states)
            total_loss += loss.cpu().detach()

            del states, actions, target_experiences, inputs, targets
        gc.collect()

        acc = 100.0 * correct / (total + EPS)
        return float(total_loss / batch_num), float(acc)


class DahaiModel(Model):
    """ActorCritic教師あり打牌用モデル
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

        self.celoss = nn.CrossEntropyLoss()
        self.mseloss = nn.MSELoss()

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
        result["train/dahai_loss"] = all_p_loss / batch_num
        result["train/value_loss"] = all_v_loss / batch_num
        result["train/reward.var"] = np.var(rewards)
        var_minus_loss = (float(np.var(rewards)) - result["train/value_loss"])
        result["train/(reward.var - value_loss)÷reward.var"] = \
            var_minus_loss / result["train/reward.var"]

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
        result["test/dahai_loss"] = all_p_loss / batch_num
        result["test/value_loss"] = all_v_loss / batch_num
        result["test/reward.var"] = np.var(rewards)
        result["test/(reward.var - value_loss)÷reward.var"] = \
            (float(np.var(rewards)) -
             result["test/value_loss"]) / result["test/reward.var"]

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

        concated = np.concatenate([
            dataset.feature.common_feature,
            dataset.feature.reach_dahai_feature[actor],
            dataset.feature.reach_dahai_feature[shimocha],
            dataset.feature.reach_dahai_feature[toimen],
            dataset.feature.reach_dahai_feature[kamicha],
        ], axis=0)
        return concated[:, :, np.newaxis]

    def _calc_label(self, dataset: Dataset):
        return Pai.str_to_id(dataset.label.next_action['pai'])

    def _calc_reward(self, dataset: Dataset):
        """
        NOTE:1000点を1としている。

        NOTE:以下を使えば割引報酬にすることも可能

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
