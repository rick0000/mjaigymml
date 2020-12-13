import time
import argparse
import subprocess
from shlex import quote
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from mjaigym.tcp.server import Server


def prepare_manue():
    build_proc = subprocess.Popen([
        "docker", "build", "-t", "manue:dev", "manue/."
    ],
        # shell=True,
        stdout=subprocess.DEVNULL,
    )
    while True:
        if build_proc.poll() is not None:
            break
        time.sleep(0.1)


def run_manue(port, player_name, room_name):
    processes = []

    for i in range(4):
        p_name = f"{quote(player_name)}{i}"
        proc = subprocess.Popen([
            "docker run --rm manue:dev /mjai/mjai/run_manue.sh " +
            f"'{p_name}' '{port}' '{quote(room_name)}'"
        ],
            shell=True,
            stdout=subprocess.DEVNULL,
        )
        processes.append(proc)

    for proc in processes:
        proc.wait()
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=48000)
    parser.add_argument("--logdir", type=str,
                        default="./experiments/output/manue_log")

    args = parser.parse_args()

    prepare_manue()

    server = Server(args.port, 128, args.logdir)
    server.run_async()

    game_num = 1000
    count = 0
    per_games = 4

    with ThreadPoolExecutor(max_workers=per_games) as executor:
        future_list = []
        for i in range(game_num):

            future = executor.submit(
                run_manue,
                args.port,
                "TestManue",
                f"room{i}"
            )
            future_list.append(future)
        futures.as_completed(fs=future_list)
        executor.shutdown()
    # wait for server mjson log output
    time.sleep(5)
