import queue
import pprint

import loggers as lgs


class Trainer():
    def __init__(
        self,
        # train_dir,
        # test_dir,
        # log_dir,
        # session_name,
        # in_on_tsumo_channels,
        # in_other_dahai_channels,
        # use_multiprocess=True,
        # udpate_interbal=64,
        # reward_discount_rate=0.99,
        # batch_size=256,
        # evaluate_per_update=5
    ):
        pass
        # self.train_dir = train_dir
        # self.test_dir = test_dir
        # self.use_multiprocess = use_multiprocess
        # self.udpate_interbal = udpate_interbal
        # self.in_on_tsumo_channels = in_on_tsumo_channels
        # self.in_other_dahai_channels = in_other_dahai_channels

        # self.mjson_path_queue = Queue(256)
        # self.experiences = Queue()

        # self.reward_discount_rate = reward_discount_rate

        # self.log_dir = log_dir
        # self.batch_size = batch_size
        # self.evaluate_per_update = evaluate_per_update
        # self.session_name = session_name
        # self.session_dir = Path(self.log_dir) / self.session_name

        # self.tfboard_logger = TensorBoardLogger(
        #     log_dir=self.log_dir, session_name=self.session_name)

    def train_loop(
            self,
            model,
            dataset_queue,
            feature_generate_process,
            train_config,
            model_config,
    ):
        game_count = 0
        dahai_update_count = 0
        # consume first observation for load
        # if load_model:
        #     agent.load(load_model, self.in_on_tsumo_channels,
        #                self.in_other_dahai_channels)

        lgs.logger_main.info("start train")
        datasets = []

        while True:
            if len(datasets) < model_config.batch_size:
                try:
                    one_mjson_dataset = dataset_queue.get(timeout=1)
                    datasets.extend(one_mjson_dataset)
                except queue.Empty:
                    if not feature_generate_process.is_alive():
                        lgs.logger_main.info(
                            "generate process finished, end train.")
                        return
            else:
                lgs.logger_main.info("filled queue, start train")
                # clear used dataset
                train_data = datasets[:model_config.batch_size]
                datasets = datasets[model_config.batch_size:]
                update_result = model.update(train_data)
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
