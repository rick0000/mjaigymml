import queue
import pprint
import random
from collections import deque
from pathlib import Path
import gc

import mlflow

import loggers as lgs
from mjaigymml.models.model import Model


class Trainer():
    def __init__(
        self,
    ):
        pass

    def train_loop(
            self,
            model: Model,
            dataset_queue,
            feature_generate_process,
            train_config,
            model_config,
            model_save_dir,
            load_model_file,
    ):
        if train_config.model_type in ["dahai", "reach", "pon", "chi", "kan"]:
            self._train(
                model,
                dataset_queue,
                feature_generate_process,
                train_config,
                model_config,
                model_save_dir,
                load_model_file,
            )
        else:
            raise NotImplementedError()

    def _train(
            self,
            model: Model,
            dataset_queue,
            feature_generate_process,
            train_config,
            model_config,
            model_save_dir,
            load_model_file,
    ):
        game_count = 0
        update_count = 0

        if load_model_file:
            model.load(load_model_file)

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
                        output_path = Path(model_save_dir) / \
                            train_config.model_type / f"{game_count}.pth"
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        model.save(output_path)
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

                mlflow.log_metrics(update_result, step=game_count)

                update_count += 1
                if update_count % 20 == 0:
                    lgs.logger_main.info("save model")
                    output_path = Path(model_save_dir) / \
                        train_config.model_type / f"{game_count}.pth"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    model.save(output_path)
