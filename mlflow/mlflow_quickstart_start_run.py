# Store a model in MLflow
# An MLflow Model is a directory that packages ML models and support files in a standard format. The directory contains:
# - An MLModel file in YAML format specifying the model’s flavor, dependencies, signature (if supplied), and  metadata;
# - The files required by the model’s flavor(s) to instantiate the model. This will often be a serialized Python object;
# - Files necessary for recreating the model’s runtime environment (for instance, a conda.yaml file);
# - Optionally, an input example
#
# When using autologging, MLflow will automatically log whatever model or models the run creates.
# You can also log a model manually by calling mlflow.{library_module_name}.log_model.
# In addition, if you wish to load the model soon, it may be convenient to output the run’s ID directly to the console.
# For that, you’ll need the object of type mlflow.ActiveRun for the current run.
# You get that object by wrapping all of your logging code in a "with mlflow.start_run() as run:" block. For example:

import mlflow
from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

with mlflow.start_run() as run:
    # Load the diabetes dataset.
    db = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

    # Create and train models.
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
    rf.fit(X_train, y_train)

    # Use the model to make predictions on the test dataset.
    predictions = rf.predict(X_test)
    # print(predictions)

    signature = infer_signature(X_test, predictions)
    mlflow.sklearn.log_model(rf, "model", signature=signature)

    print("Run ID: {}".format(run.info.run_id))
