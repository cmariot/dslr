import numpy as np
import argparse
import pandas
import yaml
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


def filter_house(y_train, house):
    if house not in ("Ravenclaw", "Hufflepuff", "Slytherin", "Gryffindor"):
        return None
    ret = np.where(y_train == house, 1, 0)
    return ret.reshape(-1, 1)


def normalize_train(x_train):

    x_norm = np.empty(x_train.shape)
    x_min = np.zeros((x_train.shape[1], 1))
    x_max = np.zeros((x_train.shape[1], 1))

    for col in range(x_train.shape[1]):
        x_min[col] = np.amin(x_train[:, col])
        x_max[col] = np.amax(x_train[:, col])
        x_norm[:, col] = \
            (x_train[:, col] - x_min[col]) / (x_max[col] - x_min[col])

    return (x_norm, x_min, x_max)


if __name__ == "__main__":

    dataset_path = parse_arguments()
    dataset = read_dataset(dataset_path)

    features = [
        "Astronomy",
        "Herbology",
        "Defense Against the Dark Arts",
        "Ancient Runes"
    ]
    target = ["Hogwarts House"]
    houses = (
        "Ravenclaw",
        "Hufflepuff",
        "Slytherin",
        "Gryffindor"
    )

    training_set = dataset[features + target].dropna()
    # split dataset into training and test sets
    x_train = training_set[features].to_numpy()
    y_train = training_set[target]
    x_norm, x_min, x_max = normalize_train(x_train)
    theta_shape = (x_norm.shape[1] + 1, 1)

    model = {}
    model["x_min"] = x_min
    model["x_max"] = x_max
    for i, house in enumerate(houses):
        print("Fiting for house", house)
        mlr = MyLogisticRegression(
            theta=np.zeros(theta_shape),
            max_iter=500,
            alpha=4.0
        )
        filtered_y = filter_house(y_train, house)
        mlr.fit_(x_norm, filtered_y)
        model[house] = mlr.theta
        mlr.plot_loss_evolution()

    with open("models.yml", "w") as file:
        yaml.dump(model, file)

    # Pas besoin de faire de prediction dans ce script,
    # ca sera fait dans logreg_predict.py.
    # Sauvegarde de ce qu'on avait fait vendredi :

    # prediction = np.empty((x_norm.shape[0], 0))
    # for house in houses:
    #   mlr = MLR(theta[i] ...)
    #   y_hat = mlr.predict_(x_norm)
    #   prediction = np.concatenate((prediction, y_hat), axis=1)
