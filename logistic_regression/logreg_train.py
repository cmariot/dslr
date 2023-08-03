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
            '--test',
            '-t',
            help='Split training set into training and test sets.',
            action='store_true',
            default=False
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
            '-m',
            help='Use the mini-batch gradient descent otpimization.',
            action='store_true',
            default=False
        )

        args = parser.parse_args()

        if (args.stochastic, args.mini_batch, args.batch).count(True) > 1:
            print("Error: you can only use one optimization method.")
            exit()

        return (
            args.dataset_path,
            args.loss_evolution,
            args.test,
            args.stochastic,
            args.mini_batch,
            True if not args.stochastic and not args.mini_batch else False
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

        # Training set = set of data that is used to train and
        # make the model learn
        m = dataset.shape[0]
        train_index_begin = 0
        train_index_end = int(m * ratios[0])
        x_train = dataset[features][train_index_begin:train_index_end]
        y_train = dataset[target][train_index_begin:train_index_end]

        # Test set = set of data that is used to test the model
        test_index_begin = train_index_end
        test_index_end = test_index_begin + int(m * ratios[1])
        x_test = dataset[features][test_index_begin:test_index_end]
        y_test = dataset[target][test_index_begin:test_index_end]

        # Return the splitted dataset as Numpy arrays
        return (x_train.to_numpy(), y_train,
                x_test.to_numpy(), y_test)

    except Exception as e:
        print("Error: Can't split the dataset")
        print(e)
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


if __name__ == "__main__":

    (dataset_path,
     display_loss_evolution,
     test_model,
     stochastic,
     mini_batch,
     batch) = parse_arguments()

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
        stochastic
    ).index(True)

    training_set = dataset[features + target]

    if test_model:
        split_ratio = (0.8, 0.2)
    else:
        split_ratio = (1, 0)

    (
        x_train,
        y_train,
        x_test,
        y_test
    ) = split_dataset(training_set, split_ratio, features, target)

    x_train = mlr.KNN_inputer(x_train, nb_neighbors=5)

    x_norm, x_min, x_max = MLR.normalize_train(x_train)

    if test_model:
        x_test = mlr.KNN_inputer(x_test, nb_neighbors=5)
        x_test = MLR.normalize_test(x_test, x_min, x_max)

    model = {}
    model["x_min"] = x_min.reshape(1, -1)
    model["x_max"] = x_max.reshape(1, -1)
    model["features"] = features

    theta_shape = (x_norm.shape[1] + 1, 1)

    def print_intro():

        options = {
            "Training mode": 'Batch' if batch
            else 'Mini-batch' if mini_batch
            else 'Stochastic',
            "Features": ", ".join(features),
            "Target": ", ".join(target),
            "Houses": ", ".join(houses),
            "Training set size": x_train.shape[0],
            "Test mode": 'Yes' if test_model else 'No',
            "Test set size": x_test.shape[0],
            "Loss evolution": 'Yes' if display_loss_evolution else 'No',
            "Save model": "Yes" if not test_model else "No"
        }

        if not test_model:
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

    print_intro()

    for i, house in enumerate(houses):

        print(f"Training model {i + 1}/4 for house {house}")

        mlr = MLR(
            theta=np.zeros(theta_shape),
            max_iter=20_000,
            learning_rate=0.1,
            stochastic=stochastic,
            mini_batch=mini_batch,
            batch_size=32
        )

        fit = (
            mlr.fit_,
            mlr.fit_mini_batch_,
            mlr.fit_stochastic_
        )

        filtered_y_train = filter_house(y_train, house)

        fit[option](
            x_norm,
            filtered_y_train,
            display_loss_evolution
        )

        y_hat = mlr.predict_(x_norm)
        y_hat = np.array([1 if x > 0.5 else 0 for x in y_hat]).reshape(-1, 1)
        mlr.one_vs_all_stats(filtered_y_train, y_hat)

        model[house] = mlr.theta

    if not test_model:
        with open("models.yml", "w") as file:
            yaml.dump(model, file)
            print("Models saved in 'models.yml' file.")

    if test_model:

        print("Testing models on test set...\n" +
              "The model will not be saved.\n")

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
