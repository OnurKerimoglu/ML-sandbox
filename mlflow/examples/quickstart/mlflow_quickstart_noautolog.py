# if you are using a library for which autolog is not yet supported, you may use key-value pairs to track:

# Name       | Used for                                      | Function call
# Parameters | Constant values (eg configuration parameters) | mlflow.log_param, mlflow.log_params
# Metrics    | Values updated during the run (eg accuracy)   | mlflow.log_metric
# Artifacts  | Files produced by the run (eg model weights)  | mlflow.log_artifacts, mlflow.log_image, mlflow.log_text

# This example demonstrates the use of these functions:

import os
from random import random, randint
from mlflow import log_metric, log_param, log_params, log_artifacts

if __name__ == "__main__":
    # Log a parameter (key-value pair)
    log_param("config_value", randint(0, 100))

    # Log a dictionary of parameters
    log_params({"param1": randint(0, 100), "param2": randint(0, 100)})

    # Log a metric; metrics can be updated throughout the run
    log_metric("accuracy", random() / 2.0)
    log_metric("accuracy", random() + 0.1)
    log_metric("accuracy", random() + 0.2)

    # Log an artifact (output file)
    # if not os.path.exists("mlruns/0/25bca33daf1c43c08234dac99c74c680/outputs"):
    #     os.makedirs("mlruns/0/25bca33daf1c43c08234dac99c74c680/outputs")
    # with open("mlruns/0/25bca33daf1c43c08234dac99c74c680/outputs/test.txt", "w") as f:
    #     f.write("hello world!")
    # log_artifacts("mlruns/0/25bca33daf1c43c08234dac99c74c680/outputs")

    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("hello world!")
    log_artifacts("outputs")
