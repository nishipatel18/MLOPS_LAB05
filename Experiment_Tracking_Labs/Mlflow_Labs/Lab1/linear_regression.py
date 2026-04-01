# ============================================
# MLflow Manual Logging - Linear Regression (Script Version)
# Dataset: California Housing (sklearn built-in)
# Run: python linear_regression.py 0.5 0.5
# ============================================

import logging
import sys
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Load California Housing dataset
    housing = fetch_california_housing(as_frame=True)
    data = housing.frame

    # Split the data into training and test sets (0.75, 0.25)
    train, test = train_test_split(data)

    train_x = train.drop(["MedHouseVal"], axis=1)
    test_x = test.drop(["MedHouseVal"], axis=1)
    train_y = train[["MedHouseVal"]]
    test_y = test[["MedHouseVal"]]

    # Accept alpha and l1_ratio from command line args
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted)

        print(f"ElasticNet model (alpha={alpha:f}, l1_ratio={l1_ratio:f}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        predictions = lr.predict(train_x)
        signature = infer_signature(train_x, predictions)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="ElasticnetHousingModel", signature=signature
            )
        else:
            mlflow.sklearn.log_model(lr, "model", signature=signature)
            