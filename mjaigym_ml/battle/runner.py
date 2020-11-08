import time
import argparse
import subprocess
import datetime
from shlex import quote
from threading import Thread
import threading

from mjaigym.tcp.server import Server


def run_manue(room_name, player_name):
    # proc = subprocess.run([
    #         f"docker build -t manue:dev manue/. && docker run --rm manue:dev /mjai/mjai/run_manue.sh '{quote(player_name)}' '{quote(room_name)}'"
    #         ], 
    #     shell=True,
    #     stdout=subprocess.DEVNULL,
    #     )
    subprocess.run(
        ["docker", "build", "-t", "manue:dev", "manue/."], 
        shell=False,
        stdout=subprocess.DEVNULL)
    proc = subprocess.run(
        ["docker", "run", "--rm", "manue:dev", "/mjai/mjai/run_manue.sh", f"{quote(player_name)}", f"{quote(room_name)}"], 
        shell=False,
        stdout=subprocess.DEVNULL)

    return proc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=48000)
    parser.add_argument("--clientlimit", type=int, default=128)
    parser.add_argument("--logdir", type=str, default="/mnt/sdc/experiments/output/manue_log")

    args = parser.parse_args()
    server = Server(args.port, args.clientlimit, args.logdir)
    server.run_async()

    

    
    game_num = 1000
    count = 0
    per_games = 16

    while game_count < game_num:
        threads = []
        count += per_games
        for room_id in range(per_games):
            print(datetime.datetime.now() , f"start game {room_id}")
            for i in range(4):
                thread = threading.Thread(target=run_manue, args=(f"room{room_id}", f"manue{i}"), daemon=True)
                thread.start()
                threads.append(thread)

        for t in threads:
            t.join()

        
    
    # wait for server mjson log output
    time.sleep(5)