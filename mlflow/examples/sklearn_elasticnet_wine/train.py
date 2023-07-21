# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties.
# In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
np.random.seed(42)


def main(pardict, test_size, expname):

    logging.info(f"Test size: {test_size}")
    data_splits = get_wine_data_splits(test_size=test_size)

    train_predict_evaluate(data_splits, pardict, test_size, expname)


def train_predict_evaluate(data_splits, pardict, test_size, expname):

    if expname != '':
        run_name = f"ElasticWineModel_{expname}"
    else:
        run_name = None  # mlflow will assign a random name to run

    with mlflow.start_run(run_name=run_name):

        train_x, test_x, train_y, test_y = data_splits

        mlflow.log_input(dataset=mlflow.data.from_pandas(train_x), context="training", tags={'size': str(1-test_size)})
        mlflow.log_input(dataset=mlflow.data.from_pandas(test_x), context="testing", tags={'size': str(test_size)})

        lr = ElasticNet(alpha=pardict['alpha'], l1_ratio=pardict['l1_ratio'], random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(pardict['alpha'], pardict['l1_ratio']))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", pardict['alpha'])
        mlflow.log_param("l1_ratio", pardict['l1_ratio'])
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        predictions = lr.predict(train_x)
        signature = infer_signature(train_x, predictions)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="ElasticnetWineModel", signature=signature
            )
        else:
            mlflow.sklearn.log_model(lr, "model", signature=signature)


def get_wine_data_splits(test_size):
    # Read the wine-quality csv file from the URL
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data, test_size=test_size)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    data_splits = (train_x, test_x, train_y, test_y)
    return data_splits


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    input_pars = {'alpha': float(sys.argv[1]) if len(sys.argv) > 1 else 0.5,
                  'l1_ratio': float(sys.argv[2]) if len(sys.argv) > 2 else 0.5}
    input_testsize = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
    input_expname = sys.argv[4] if len(sys.argv) > 4 else ''

    main(pardict=input_pars, test_size=input_testsize, expname=input_expname)
