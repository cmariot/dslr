import numpy as np
import argparse
import pandas
import yaml
from my_logistic_regression import MyLogisticRegression as MLR
# from logic_reg import MyLogisticRegression


def parse_arguments() -> tuple:
    """
    Parse arguments of the program.
    describe.py takes a dataset_path as argument.
    """
    try:
        parser = argparse.ArgumentParser(
            prog="logreg_train.py",
            description="This program takes a dataset path as argument. " +
            "It trains the models and save the results in 'model.yml' file."
        )
        parser.add_argument(
            'dataset_path',
            type=str,
            help='Path to the training dataset.',
            default="../datasets/dataset_train.csv"
        )
        parser.add_argument(
            '--loss_evolution',
            '-l',
            help='Display loss evolution during training.',
            action='store_true',
            default=False
        )
        args = parser.parse_args()
        return (
            args.dataset_path,
            args.loss_evolution
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


def filter_house(y_train, house):
    try:
        if house not in ("Ravenclaw", "Hufflepuff", "Slytherin", "Gryffindor"):
            return None
        ret = np.where(y_train == house, 1, 0)
        return ret.reshape(-1, 1)
    except Exception as e:
        print("Error filtering house: ", e)
        exit()


if __name__ == "__main__":

    (dataset_path, display_loss_evolution) = parse_arguments()
    dataset = read_dataset(dataset_path)

    features = [
        "Arithmancy",
        "Astronomy",
        "Herbology",
        "Defense Against the Dark Arts",
        "Divination",
        "Muggle Studies",
        "Ancient Runes",
        "History of Magic",
        "Transfiguration",
        "Potions",
        "Care of Magical Creatures",
        "Charms",
        "Flying",
    ]

    features = [
        "Astronomy",
        "Defense Against the Dark Arts",
        "Ancient Runes",
        "Herbology",
        "Charms",
        "Divination",
        "Muggle Studies",

    ]

    polynomial_degree = 1
    target = ["Hogwarts House"]
    houses = (
        "Ravenclaw",
        "Slytherin",
        "Gryffindor",
        "Hufflepuff"
    )

    training_set = dataset[features + target].dropna()
    x_train = training_set[features].to_numpy()
    y_train = training_set[target]
    x_train_poly = \
        MLR.add_polynomial_features(x_train, polynomial_degree)
    x_train_norm, x_min, x_max = \
        MLR.normalize_train(x_train_poly)
    model = {}
    model["x_min"] = x_min.reshape(1, -1)
    model["x_max"] = x_max.reshape(1, -1)
    model["polynomial_degree"] = polynomial_degree
    model["features"] = features
    theta_shape = (x_train_norm.shape[1] + 1, 1)

    for i, house in enumerate(houses):
        print(f"Training model {i + 1}/4 for house {house}")
        mlr = MLR(
            theta=np.zeros(theta_shape),
            max_iter=100000,
            alpha=0.01,
            penality=None,
            lambda_=0.0,
        )
        filtered_y_train = filter_house(y_train, house)
        mlr.fit_(
            x_train_norm,
            filtered_y_train,
            display_loss_evolution
        )
        model[house] = mlr.theta

    with open("models.yml", "w") as file:
        yaml.dump(model, file)
        print("Models saved in 'models.yml' file.\n")

    prediction = np.empty((x_train_norm.shape[0], 0))
    for house in houses:
        mlr.theta = model[house]
        y_hat = mlr.predict_(x_train_norm)
        prediction = np.concatenate((prediction, y_hat), axis=1)

    # Argmax sur les predictions pour trouver la maison la plus probable
    # pour chaque ligne du dataset d'entrainement
    y_hat = np.argmax(prediction, axis=1)
    # On remplace les indices par les noms des maisons
    y_hat = np.array([houses[i] for i in y_hat])
    # On compare les predictions avec les vraies valeurs
    y_train = y_train.to_numpy().reshape(-1)

    # Confusion matrix
    mlr.confusion_matrix_(
        y_train,
        y_hat,
        labels=houses,
        df_option=True,
        display=True
    )

    print("\nAccuracy on training set:")
    accuracy = mlr.accuracy_score_(y_hat, y_train)
    print(accuracy * 100, "%")
