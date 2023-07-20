# To load and run a model stored in a previous run, you can use the mlflow.{library_module_name}.load_model function.
# You’ll need the run ID of the run that logged the model.

import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# model = mlflow.sklearn.load_model("runs:/88a4ff2b972240faaffeba2e8990ecab/model")
model = mlflow.sklearn.load_model("runs:/25bca33daf1c43c08234dac99c74c680/model")

predictions = model.predict(X_test)
print(predictions)
