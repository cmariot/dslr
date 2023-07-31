import numpy as np
import argparse
import pandas
import yaml
from my_logistic_regression import MyLogisticRegression as MLR


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

    (test_path, model_path) = parse_arguments()
    model = get_model(model_path)
    data_test = read_dataset(test_path)
    truth = read_dataset("../datasets/dataset_truth.csv")
    truth = truth["Hogwarts House"].to_numpy()

    houses = (
        "Ravenclaw",
        "Slytherin",
        "Gryffindor",
        "Hufflepuff"
    )

    x_test = data_test[model["features"]].to_numpy()
    x_test = MLR.add_polynomial_features(x_test, model["degree"])
    x_test = normalize_test(x_test, model["x_min"], model["x_max"])

    mlr = MLR()
    prediction = np.empty((x_test.shape[0], 0))
    for house in houses:
        mlr.theta = model[house]
        y_hat = mlr.predict_(x_test)
        prediction = np.concatenate((prediction, y_hat), axis=1)

    y_hat = np.argmax(prediction, axis=1)
    # On remplace les indices par les noms des maisons
    y_hat = np.array([houses[i] for i in y_hat])

    # print(y_hat.shape)
    # print(truth.shape)

    mlr.confusion_matrix_(
        y_true=truth,
        y_hat=y_hat,
        labels=houses,
        df_option=True,
        display=True
    )

    print(f"\nAccurency: {mlr.accuracy_score_(y_hat, truth) * 100} %")
