import os
from random import random, randint
from mlflow import log_metric, log_param, log_artifacts
import mlflow

if __name__ == "__main__":
    mlflow.set_experiment("Battle")
    # Log a parameter (key-v]alue pair)
    log_param("param1", randint(0, 100))

    # Log a metric; metrics can be updated throughout the run
    for _ in range(10000):
        log_metric("foo", random())
        log_metric("foo", random() + 1)
        log_metric("foo", random() + 2)

        log_metric("bar", random())
        log_metric("baz", random() + 1)

    # Log an artifact (output file)
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("hello world!")
    log_artifacts("outputs")
