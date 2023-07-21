import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from train import eval_metrics

logged_model = 'runs:/aed6184b60ff4efdad6beb167d188cde/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.

# Read the wine-quality csv file from the URL
csv_url = ("https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv")
data = pd.read_csv(csv_url, sep=";")

# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(data)

# The predicted column is "quality" which is a scalar from [3, 9]
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]

predicted_qualities = loaded_model.predict(pd.DataFrame(test_x))

(rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

print("Elasticnet model ")
print("  RMSE: %s" % rmse)
print("  MAE: %s" % mae)
print("  R2: %s" % r2)
