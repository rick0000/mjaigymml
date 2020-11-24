import datetime
import numpy as np


def run():
    save()
    # load()


def save():
    a1 = np.random.randint(low=0, high=2, size=(
        500, 1000, 34), dtype='uint8')

    print(a1.nbytes / 1_000_000)  # MB

    print(datetime.datetime.now(), "start save")
    np.save("a1.npy", a1)
    print(datetime.datetime.now(), "end save")

    print(datetime.datetime.now(), "start save npz")
    np.savez("a1npz", a1)
    print(datetime.datetime.now(), "end save")

    print(datetime.datetime.now(), "start save npz compressed")
    np.savez_compressed("a1npzc", a1)
    print(datetime.datetime.now(), "end save")

    del a1


def load():
    repeat_num = 1

    # print(datetime.datetime.now(), "start load")
    # a2_length = 0
    # for _ in range(repeat_num):
    #     a2 = np.load("a1.npy")
    #     a2_length += len(a2)
    # print(datetime.datetime.now(), "end load")

    print(datetime.datetime.now(), "start load npz")
    a2_length = 0
    for _ in range(repeat_num):
        a2 = np.load("a1npz.npz")['arr_0']
        a2_length += len(a2)
    print(datetime.datetime.now(), "end load npz")

    print(datetime.datetime.now(), "start load npz compressed")
    a2_length = 0
    for _ in range(repeat_num):
        a2 = np.load("a1npzc.npz")['arr_0']
        a2_length += len(a2)
    print(datetime.datetime.now(), "end load npz")


if __name__ == "__main__":
    run()
