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
            '-d', '--dataset_path',
            dest='dataset_path',
            type=str,
            help='Path to the training dataset.',
            default="../datasets/dataset_test.csv",
        )
        parser.add_argument(
            '-m', '--model_path',
            dest='model_path',
            type=str,
            help='Please specify the model.yml',
            default="./models.yml",
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
        if model is None:
            raise Exception("The model is empty.")
        required_keys = [
            "features",
            "x_min",
            "x_max",
            "Ravenclaw",
            "Slytherin",
            "Gryffindor",
            "Hufflepuff"
        ]
        for key in required_keys:
            if key not in model:
                raise Exception("The model is missing " + key + ".")
        return model
    except FileNotFoundError:
        print("Error: model not found, please use logreg_train.py before.")
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
        if dataset.empty:
            raise Exception("The dataset is empty.")
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


def predict(x_norm, model, mlr):
    try:
        houses = (
            "Ravenclaw",
            "Slytherin",
            "Gryffindor",
            "Hufflepuff"
        )
        prediction = np.empty((x_norm.shape[0], 0))
        for house in houses:
            mlr.theta = model[house]
            y_hat = mlr.predict_(x_norm)
            prediction = np.concatenate((prediction, y_hat), axis=1)
        y_hat = np.argmax(prediction, axis=1)
        return np.array([houses[i] for i in y_hat]), prediction
    except Exception as e:
        print("Error predicting: ", e)
        exit()


def print_intro():
    print("""
 _                                                 _ _      _
| | ___   __ _ _ __ ___  __ _   _ __  _ __ ___  __| (_) ___| |_
| |/ _ \\ / _` | '__/ _ \\/ _` | | '_ \\| '__/ _ \\/ _` | |/ __| __|
| | (_) | (_| | | |  __/ (_| | | |_) | | |  __/ (_| | | (__| |_
|_|\\___/ \\__, |_|  \\___|\\__, | | .__/|_|  \\___|\\__,_|_|\\___|\\__|
         |___/          |___/  |_|


Probabilities of each house and assigned house for each student:
""")


def save_prediction(y_hat, path):
    try:
        df = pandas.DataFrame(
            data=y_hat.reshape(-1, 1),
            columns=["Hogwarts House"]
        )
        with open(path, "w") as file:
            df.to_csv(
                path_or_buf=file,
                index=True,
                index_label="Index",
                header=True
            )
            print("Prediction saved in 'houses.csv' file.")
    except Exception as e:
        print("Error saving prediction: ", e)
        exit()


if __name__ == '__main__':

    try:

        test_path, model_path = parse_arguments()
        model = get_model(model_path)
        test_dataset = read_dataset(test_path)

        mlr = MLR()
        features = model["features"]
        x_test = test_dataset[features].to_numpy()
        x_test = mlr.knn_imputer(x_test, nb_neighbors=5)
        x_norm = normalize_test(x_test, model["x_min"], model["x_max"])

        y_hat, proba = predict(x_norm, model, mlr)

        proba = np.concatenate(
            (np.round(proba, decimals=4), y_hat.reshape(-1, 1)),
            axis=1
        )

        df = pandas.DataFrame(
            data=proba,
            columns=[
                "Ravenclaw",
                "Slytherin",
                "Gryffindor",
                "Hufflepuff",
                "  Predicted House"
            ]
        )

        print_intro()
        with pandas.option_context("display.max_rows", None):
            print(df, "\n")

        save_prediction(y_hat, "houses.csv")

    except Exception as e:
        print("Error: ", e)
        exit()
