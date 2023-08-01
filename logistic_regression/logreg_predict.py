import numpy as np
import argparse
import pandas
import yaml
from my_logistic_regression import MyLogisticRegression as MLR
from sklearn.impute import KNNImputer


def parse_arguments() -> tuple:
    """
    Parse arguments of the program.
    describe.py takes a dataset_path as argument.
    """
    try:
        parser = argparse.ArgumentParser(
            prog="logreg_predict.py",
            description="This program takes a dataset path as argument " +
            "and a model. It predict the result of the models passed as " +
            "parameter on the dataset and save the results in " +
            "'houses.csv' file."
        )
        parser.add_argument(
            'dataset_path',
            type=str,
            help='Path to the training dataset.',
            default="../datasets/dataset_train.csv"
        )
        parser.add_argument(
            'model_path',
            type=str,
            help='Please specify the model.yml',
            default="./model.yml"
        )
        args = parser.parse_args()
        return (
            args.dataset_path,
            args.model_path
        )

    except Exception as e:
        print("Error parsing arguments: ", e)
        exit()


def get_model(model_path: str) -> dict:
    """
    Read the model from the given path,
    returned as a dict.
    """
    try:
        with open(model_path, "r") as file:
            model = yaml.load(file, Loader=yaml.loader.UnsafeLoader)
        return model
    except FileNotFoundError:
        print("Error: model not found.")
        exit()
    except Exception as e:
        print("Error reading model: ", e)
        exit()


def read_dataset(dataset_path: str) -> pandas.DataFrame:
    """
    Read the dataset from the given path,
    returned as a pandas DataFrame.
    """
    try:
        dataset = pandas.read_csv(dataset_path)
        return dataset
    except FileNotFoundError:
        print("Error: dataset not found.")
        exit()
    except Exception as e:
        print("Error reading dataset: ", e)
        exit()


def normalize_test(x_test, x_min, x_max):
    try:
        x_norm = (x_test - x_min) / (x_max - x_min)
        return x_norm
    except Exception as e:
        print("Error normalizing test set: ", e)
        exit()


if __name__ == '__main__':

    test_path, model_path = parse_arguments()
    model = get_model(model_path)
    data_test = read_dataset(test_path)
    mlr = MLR()

    houses = (
        "Ravenclaw",
        "Slytherin",
        "Gryffindor",
        "Hufflepuff"
    )
    features = model["features"]



    x_test = data_test[features].to_numpy()
    # print(data_test[features])

    # print(x_test.shape)
    # exit()
    # Missing values need to be replaced.
    # Process called imputation : Replace NaN with new value.
    # We can replace with 0, mean, mode, median ...
    # But the best is tp use knn imputation
    # -> chose x nearest neighbors, take the mean feature value of them.

    # imputer = KNNImputer(n_neighbors=5)
    # x_test = imputer.fit_transform(x_test)

    x_test = mlr.KNN_inputer(x_test)

    x_norm = normalize_test(x_test, model["x_min"], model["x_max"])

    

    prediction = np.empty((x_norm.shape[0], 0))
    for house in houses:
        mlr.theta = model[house]
        y_hat = mlr.predict_(x_norm)
        prediction = np.concatenate((prediction, y_hat), axis=1)

    y_hat = np.argmax(prediction, axis=1)
    y_hat = np.array([houses[i] for i in y_hat])

    with open("houses.csv", "w") as file:
        pandas.DataFrame.to_csv(
            pandas.DataFrame(
                data=y_hat.reshape(-1, 1),
                columns=["Hogwarts House"]
            ),
            file,
            index=True,
            index_label="Index",
            header=True
        )

    truth = read_dataset("../datasets/dataset_truth.csv")
    truth = truth["Hogwarts House"].to_numpy()

    mlr.confusion_matrix_(
        y_true=truth,
        y_hat=y_hat,
        labels=houses,
        df_option=True,
        display=True
    )

    print(f"\nAccuracy: {mlr.accuracy_score_(y_hat, truth) * 100:.2f} %")
