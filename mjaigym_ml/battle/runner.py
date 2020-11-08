import time
import argparse
import subprocess
from shlex import quote
from threading import Thread
import threading

from mjaigym.tcp.server import Server


def run_manue(room_name, player_name):
    proc = subprocess.run([
            f"docker build -t manue:dev manue/. && docker run --rm manue:dev /mjai/mjai/run_manue.sh '{quote(player_name)}' '{quote(room_name)}'"
            ], 
        shell=True,
        stdout=subprocess.DEVNULL,
        )
    return proc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=48000)
    parser.add_argument("--clientlimit", type=int, default=32)
    parser.add_argument("--logdir", type=str, default="./tcplog")

    args = parser.parse_args()
    server = Server(args.port, args.clientlimit, args.logdir)
    server.run_async()

    room = "room0"

    threads = []
    for i in range(4):
        thread = threading.Thread(target=run_manue, args=(room, f"manue{i}"), daemon=True)
        thread.start()
        threads.append(thread)

    for t in threads:
        t.join()
    
    # wait for server mjson log output
    time.sleep(5)