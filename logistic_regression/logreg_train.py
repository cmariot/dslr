from os import get_terminal_size
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
            prog="logreg_train.py",
            description="This program takes a dataset path as argument. " +
            "It trains the models and save the results in 'models.yml' file."
        )
        parser.add_argument(
            '--dataset_path',
            dest='dataset_path',
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
            '-a',
            '--learning-rate',
            dest='learning_rate',
            type=float,
            nargs='?',
            help='Value of the learning_rate (alpha) used for training.',
            default=0.1
        )
        parser.add_argument(
            '-i',
            '--nb-iterations',
            dest='iterations',
            type=int,
            nargs='?',
            help='Number of iterations used for training.',
            default=20_000
        )
        parser.add_argument(
            '--test_ratio',
            dest='ratio',
            type=float,
            nargs='?',
            help='Ratio of the dataset to use for training.',
            default=0.0
        )
        parser.add_argument(
            '--batch',
            '-b',
            help='Use the batch gradient descent.',
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
            '-mb',
            help='Use the mini-batch gradient descent otpimization.',
            action='store_true',
            default=False
        )

        parser.add_argument(
            '--multi-stochastic',
            '-ms',
            help='Use the gradient descent on one randomly choosen feature for each iteration.',
            action='store_true',
            default=False
        )

        args = parser.parse_args()

        if (args.stochastic,
            args.mini_batch,
            args.batch,
                args.multi_stochastic).count(True) > 1:
            print("Error: you can only use one optimization method.")
            exit()

        ratio = args.ratio

        if ratio is None:
            ratio = 0.5
        elif ratio < 0.0 or ratio > 1.0:
            print("Error: ratio must be between 0 and 1.")
            exit()

        test = True if ratio != 0.0 else False

        if args.learning_rate <= 0.0 or args.learning_rate > 1.0:
            print("Error: learning rate must be greater than 0.")
            exit()

        if args.iterations <= 0:
            print("Error: number of iterations must be greater than 0.")
            exit()

        return (
            args.dataset_path,
            args.loss_evolution,
            ratio,
            test,
            args.stochastic,
            args.mini_batch,
            args.multi_stochastic,
            True if not args.stochastic and not args.mini_batch and not args.multi_stochastic else False,
            args.learning_rate,
            args.iterations
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
        if dataset.empty:
            print("The dataset is empty.")
            return None
        return dataset
    except FileNotFoundError:
        print("Error: dataset not found.")
        exit()
    except Exception as e:
        print("Error reading dataset: ", e)
        exit()


def split_dataset(dataset: pandas.DataFrame,
                  ratios: tuple,
                  features: list,
                  target: list) -> tuple:

    try:
        # Shuffle the dataset
        dataset = dataset.sample(frac=1)

        y = dataset[target]
        x = dataset[features].to_numpy()
        mlr = MLR()
        full_x = mlr.knn_imputer(x, nb_neighbors=5)

        m = dataset.shape[0]
        train_index_begin = 0
        train_index_end = int(m * ratios[0])
        x_train = full_x[train_index_begin:train_index_end]
        y_train = y[train_index_begin:train_index_end]

        test_index_begin = train_index_end
        x_test = full_x[test_index_begin:]
        y_test = y[test_index_begin:]

        return (x_train, y_train,
                x_test, y_test)

    except Exception:
        print("Error: Can't split the dataset")
        exit(1)


def filter_house(y_train, house):
    try:
        if house not in ("Ravenclaw", "Hufflepuff", "Slytherin", "Gryffindor"):
            return None
        ret = np.where(y_train == house, 1, 0)
        return ret.reshape(-1, 1)
    except Exception as e:
        print("Error filtering house: ", e)
        exit()


def print_intro():

    options = {
        "Training mode": 'Batch' if batch
        else 'Mini-batch' if mini_batch
        else 'Stochastic' if stochastic
        else 'Multi-stochastic',
        "Features": ", ".join(features),
        "Target": ", ".join(target),
        "Houses": ", ".join(houses),
        "Training set size": x_train.shape[0],
        "Test mode": 'Yes' if test else 'No',
        "Test set size": x_test.shape[0],
        "Loss evolution": 'Yes' if display_loss_evolution else 'No',
        "Save model": "Yes" if not test else "No"
    }

    if not test:
        del options["Test set size"]

    df_options = pandas.DataFrame(
        data=options.values(),
        index=options.keys(),
        columns=[""]

    )
    print("""
\t\t _                              _             _
\t\t| | ___   __ _ _ __ ___  __ _  | |_ _ __ __ _(_)_ __
\t\t| |/ _ \\ / _` | '__/ _ \\/ _` | | __| '__/ _` | | '_ \\
\t\t| | (_) | (_| | | |  __/ (_| | | |_| | | (_| | | | | |
\t\t|_|\\___/ \\__, |_|  \\___|\\__, |  \\__|_|  \\__,_|_|_| |_|
\t\t         |___/          |___/

""")

    with pandas.option_context(
        'display.max_columns', None,
        'display.width', get_terminal_size().columns,
        'display.max_colwidth', None,
    ):
        print(df_options, "\n\n")


if __name__ == "__main__":

    (dataset_path,
     display_loss_evolution,
     split_ratio,
     test,
     stochastic,
     mini_batch,
     multi_stochastic,
     batch,
     learning_rate,
     iterations) = parse_arguments()

    dataset = read_dataset(dataset_path)
    if dataset is None:
        exit()
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

    option = (
        batch,
        mini_batch,
        stochastic,
        multi_stochastic
    ).index(True)

    training_set = dataset[features + target]

    if test:
        split_ratio = (1. - split_ratio, split_ratio)
    else:
        split_ratio = (1., 0.)

    (
        x_train,
        y_train,
        x_test,
        y_test
    ) = split_dataset(training_set, split_ratio, features, target)

    x_norm, x_min, x_max = MLR.normalize_train(x_train)
    if test:
        x_test = MLR.normalize_test(x_test, x_min, x_max)

    model = {}
    model["x_min"] = x_min.reshape(1, -1)
    model["x_max"] = x_max.reshape(1, -1)
    model["features"] = features

    theta_shape = (x_norm.shape[1] + 1, 1)

    print_intro()

    for i, house in enumerate(houses):

        print(f"Training model {i + 1}/4 for house {house} :\n")

        mlr = MLR(
            theta=np.zeros(theta_shape),
            max_iter=iterations,
            learning_rate=learning_rate,
            stochastic=stochastic,
            mini_batch=mini_batch,
            batch_size=32
        )

        fit = (
            mlr.fit_,
            mlr.fit_mini_batch_,
            mlr.fit_stochastic_,
            mlr.fit_multi_stochastic_
        )

        filtered_y_train = filter_house(y_train, house)

        fit[option](
            x_norm,
            filtered_y_train,
            display_loss_evolution
        )

        if test:
            y_hat = mlr.predict_(x_test)
            y_stat = filter_house(y_test, house)
        else:
            y_hat = mlr.predict_(x_norm)
            y_stat = filtered_y_train
        y_hat = np.array([1 if x > 0.5 else 0 for x in y_hat]).reshape(-1, 1)

        mlr.one_vs_all_stats(y_stat, y_hat)

        model[house] = mlr.theta

    if not test:
        with open("models.yml", "w") as file:
            yaml.dump(model, file)
            print("Models saved in 'models.yml' file.")

    if not test:
        x_test = x_norm
        y_test = y_train
        test_mode = "train"
    else:
        test_mode = "test"

    print(f"Testing models on {test_mode} set...\n" +
          "The model will not be saved.\n")

    for house in houses:
        print(f"Model for {house} :")
        print(model[house], "\n")

    prediction = np.empty((x_test.shape[0], 0))
    for house in houses:
        mlr.theta = model[house]
        y_hat = mlr.predict_(x_test)
        prediction = np.concatenate((prediction, y_hat), axis=1)

    y_hat = np.argmax(prediction, axis=1)
    y_hat = np.array([houses[i] for i in y_hat])
    y_test = y_test.to_numpy().reshape(-1)

    mlr.confusion_matrix_(
        y_test,
        y_hat,
        labels=houses,
        df_option=True,
        display=True
    )

    accuracy = mlr.accuracy_score_(y_hat, y_test) * 100
    print(f"\nAccuracy on test set: {accuracy:.2f} %")
