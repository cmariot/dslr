import numpy as np
import math as mat
import argparse
import pandas

from logic_reg import MyLogisticRegression

def parse_arguments() -> tuple:
    """
    Parse arguments of the program.
    describe.py takes a dataset_path as argument.
    """
    try:
        parser = argparse.ArgumentParser(
            prog="describe",
            description="This program takes a dataset path as argument. " +
            "It displays informations for all numerical features."
        )
        parser.add_argument('dataset_path')
        args = parser.parse_args()
        return (
            args.dataset_path
        )

    except Exception as e:
        print("Error parsing arguments: ", e)
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


def select_columns(dataset: pandas.DataFrame) -> pandas.DataFrame:
    """
    Describe display numerical features metrics.
    """
    try:
        numerical_dataset = dataset.select_dtypes(include="number")
        without_index = numerical_dataset.drop("Index", axis='columns')
        columns = without_index.columns
        return (
            without_index,
            columns
        )
    except KeyError:
        # When the dataset does not contain an "Index" column,
        # the drop method will raise a KeyError.
        return (
            numerical_dataset,
            numerical_dataset.columns
        )
    except Exception as e:
        print("Error selecting columns: ", e)
        exit()

def filter_house(Y_train, house):
    if not house in ["Ravenclaw" ,"Hufflepuff","Slytherin","Gryffindor"]:
        return None
    ret = np.where(Y_train == house, 1, 0)
    return ret


def normalize_train(X_train):

    ret = X_train.copy()
    for col in range(X_train.shape[1]):
        amin = np.amin(X_train[:, col])
        amax = np.amax(X_train[:, col])
        ret[:,col] = (X_train[:,col] - amin) / (amax - amin)
    return ret
    

if __name__ == "__main__":

    dataset_path = parse_arguments()
    entire_dataset = read_dataset(dataset_path)

    data_used = entire_dataset[["Hogwarts House", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Ancient Runes"]].dropna()

    X_train = data_used[["Astronomy", "Herbology", "Defense Against the Dark Arts", "Ancient Runes"]].to_numpy()

    Y_train = data_used["Hogwarts House"]

    houses = ["Ravenclaw" ,"Hufflepuff","Slytherin","Gryffindor"]

    mod = {}

    X_norm = normalize_train(X_train)

    #prediction = np.full((X_train.shape[0], 4), 0)s
    prediction = np.full((1470, 1), 0)
    print(prediction.shape)

    for i, house in enumerate(houses):
        print("Fiting for house", house)
        mlr = MyLogisticRegression(theta=np.zeros((5, 1)), max_iter=1000, alpha=0.1)
        mlr.fit_(X_norm, filter_house(Y_train, house).reshape(-1, 1))
        mod[house] = mlr.theta
        prediction = np.concatenate((prediction, mlr.predict_(X_norm).reshape(-1, 1)), axis=1)
        #prediction[ : , i] = mlr.predict_(X_norm).reshape(-1, 1)

    print(prediction)
