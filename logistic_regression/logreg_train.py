import numpy as np
import argparse
import pandas
from sklearn.impute import KNNImputer
import yaml
from my_logistic_regression import MyLogisticRegression as MLR
# from logic_reg import MyLogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt


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

        parser.add_argument(
            '--stochastic',
            '-s',
            help='Use the stochastic gradient descent optimization.',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--mini-batch',
            '-m',
            help='Use the mini-batch gradient descent otpimization.',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--batch',
            '-b',
            help='Use the batch gradient descent otpimization.',
            action='store_true',
            default=False
        )

        args = parser.parse_args()

        opti = [i for i in (args.stochastic, args.mini_batch, args.batch) if i is True]
        if len(opti) > 1:
            print("Error: you can only use one optimization method.")
            exit()

        return (
            args.dataset_path,
            args.loss_evolution,
            args.stochastic,
            args.mini_batch,
            args.batch
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

    (dataset_path,
    display_loss_evolution,
    stochastic,
    mini_batch,
    batch) = parse_arguments()

    dataset = read_dataset(dataset_path)
    mlr = MLR()

    features = [
        "Astronomy",
        "Herbology",
        "Defense Against the Dark Arts",
        "Ancient Runes",
    ]

    target = ["Hogwarts House"]
    houses = (
        "Ravenclaw",
        "Slytherin",
        "Gryffindor",
        "Hufflepuff"
    )

    training_set = dataset[features + target]

    x_train = training_set[features].to_numpy()
    y_train = training_set[target]

    # Pair plot to see the distribution of the features
    # sns.pairplot(training_set, hue="Hogwarts House")
    # plt.show()

    x_train = mlr.KNN_inputer(x_train, nb_neighbors=5)

    x_norm, x_min, x_max = MLR.normalize_train(x_train)

    model = {}
    model["x_min"] = x_min.reshape(1, -1)
    model["x_max"] = x_max.reshape(1, -1)
    model["x_norm"] = x_norm
    model["features"] = features

    theta_shape = (x_norm.shape[1] + 1, 1)

    for i, house in enumerate(houses):
        print(f"Training model {i + 1}/4 for house {house}")
        mlr = MLR(
            theta=np.zeros(theta_shape),
            max_iter=20_000,
            alpha=0.1,
            penality=None,
            lambda_=0.0,
            stochastic=stochastic,
            mini_batch=mini_batch,
            batch=batch
        )
        filtered_y_train = filter_house(y_train, house)

        different_fit = [
            (stochastic, mlr.fit_stochastic_),
            # (mini_batch: mlr.fit_mini_batch_),
            (batch, mlr.fit_batch_),
            (True, mlr.fit_)
        ]

        # First element of different_fit that is True at index 0
        fit_to_use = next(
            (fit for fit in different_fit if fit[0] is True),
            (False, None)
        )[1]



        fit_to_use(
            x_norm,
            filtered_y_train,
            display_loss_evolution
        )

    

        y_hat = mlr.predict_(x_norm)
        y_hat = np.array([1 if x > 0.5 else 0 for x in y_hat]).reshape(-1, 1)
        print (y_hat.shape)
        print("and")
        print(filtered_y_train.shape)
        mlr.one_vs_all_stats(filtered_y_train, y_hat)

        model[house] = mlr.theta

    with open("models.yml", "w") as file:
        yaml.dump(model, file)
        print("Models saved in 'models.yml' file.\n")

    prediction = np.empty((x_norm.shape[0], 0))
    for house in houses:
        mlr.theta = model[house]
        y_hat = mlr.predict_(x_norm)
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
