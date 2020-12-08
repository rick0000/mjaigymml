import queue
import pprint
import random
from collections import deque

import loggers as lgs


class Trainer():
    def __init__(
        self,
    ):
        pass

    def train_loop(
            self,
            model,
            dataset_queue,
            feature_generate_process,
            train_config,
            model_config,
    ):
        if train_config.model_type == "dahai":
            self._train_dahai(
                model,
                dataset_queue,
                feature_generate_process,
                train_config,
                model_config,
            )
        elif train_config.model_type == "pon":
            self._train_dahai(
                model,
                dataset_queue,
                feature_generate_process,
                train_config,
                model_config,
            )

    def _train_dahai(
            self,
            model,
            dataset_queue,
            feature_generate_process,
            train_config,
            model_config,
    ):
        game_count = 0

        # consume first observation for load
        # if load_model:
        #     agent.load(load_model, self.in_on_tsumo_channels,
        #                self.in_other_dahai_channels)

        lgs.logger_main.info("start train")
        datasets = []
        chunk_length = model_config.batch_size * 10
        replay_buffer_length = min(10000, chunk_length * 10)
        replay_buffer = deque(maxlen=replay_buffer_length)
        while True:

            if len(datasets) < chunk_length:
                try:
                    one_mjson_dataset = dataset_queue.get(timeout=1)
                    game_count += 1
                    datasets.extend(one_mjson_dataset)
                except queue.Empty:
                    if not feature_generate_process.is_alive():
                        lgs.logger_main.info(
                            "generate process finished, end train.")
                        return
            else:
                lgs.logger_main.info("filled queue, start train")
                # clear used dataset
                train_data = datasets[:chunk_length]
                datasets = datasets[chunk_length:]

                replay_buffer.extend(train_data)
                if len(replay_buffer) < replay_buffer.maxlen:
                    lgs.logger_main.info("replay_buffer not full, continue")
                    continue
                replay_buffer = list(replay_buffer)
                random.shuffle(replay_buffer)

                train_data = replay_buffer[:chunk_length]
                replay_buffer = replay_buffer[chunk_length:]
                replay_buffer = deque(
                    replay_buffer, maxlen=replay_buffer_length)

                update_result = model.update(train_data)
                lgs.logger_main.info(f"game count:{game_count}")
                pprint.pprint(update_result)

    # def train_dahai(
    #         self,
    #         dahai_state_action_rewards: StateActionRewards,
    #         agent: MjAgent,
    #         game_count: int):
    #     result = agent.update_dahai(dahai_state_action_rewards)
    #     lgs.logger_main.info("update result")
    #     for key, value in result.items():
    #         lgs.logger_main.info(f"{key}:{value:.03f}")
    #     for key, value in result.items():
    #         self.tfboard_logger.write(key, value, game_count)
    #     rs = np.array([r[2] for r in dahai_state_action_rewards])
    #     lgs.logger_main.info(
    #         f"rewards var:{np.var(rs):.03f}, max:{rs.max():.03f}, min:{rs.min():.03f}, mean:{rs.mean():.03f}")

    # def evaluate_dahai(
    #         self,
    #         dahai_state_action_rewards: StateActionRewards,
    #         agent: MjAgent,
    #         game_count: int):
    #     lgs.logger_main.info("-------------------------------------")
    #     lgs.logger_main.info("test result")
    #     result = agent.evaluate_dahai(dahai_state_action_rewards)
    #     for key, value in result.items():
    #         lgs.logger_main.info(f"{key}:{value:.03f}")

    #     for key, value in result.items():
    #         self.tfboard_logger.write(key, value, game_count)
    #     rs = np.array([r[2] for r in dahai_state_action_rewards])
    #     lgs.logger_main.info(
    #         f"rewards var:{np.var(rs):.03f}, max:{rs.max():.03f}, min:{rs.min():.03f}, mean:{rs.mean():.03f}")
    #     lgs.logger_main.info("-------------------------------------")
