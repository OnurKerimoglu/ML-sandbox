from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import mlflow
# import mlflow.sklearn
mlflow.sklearn.autolog()


def main():
    input_data = load_iris_data()
    params_combs = {
         'ln3md2': {'leaf_nodes': 3, 'max_depth': 2},
         'ln3md3': {'leaf_nodes': 3, 'max_depth': 3},
         'ln4md2': {'leaf_nodes': 4, 'max_depth': 2}}

    for runname, params in params_combs.items():
        model, test_metrics = train_predict_evaluate_dtree(input_data, params, runname)

        print(f'Exp:{runname}: {test_metrics}')


def load_iris_data():
    data = load_iris()
    x = data.data
    y = data.target
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.25,
                                                        random_state=10)
    input_data = (x_train, x_test, y_train, y_test)
    return input_data


def train_predict_evaluate_dtree(input_data, params, runname):
    with mlflow.start_run(run_name=f"Decision Tree Classifier Exp: {runname}"):
        x_train, x_test, y_train, y_test = input_data
        clf = DecisionTreeClassifier(random_state=42,
                                     max_leaf_nodes=params['leaf_nodes'],
                                     max_depth=params['max_depth'])
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        test_accuracy = metrics.accuracy_score(y_test, y_pred)
        test_f1_score = metrics.f1_score(y_test, y_pred, average='weighted')
        test_metrics = {'test_accuracy': test_accuracy,
                        'test_f1_score': test_f1_score}

        mlflow.log_metric('test_accuracy', test_accuracy)
        mlflow.log_metric('test_f1_score', test_f1_score)
    return clf, test_metrics


if __name__ == '__main__':
    main()
