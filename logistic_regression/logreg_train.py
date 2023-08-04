import argparse
import numpy as np
import pandas
import yaml
from my_logistic_regression import MyLogisticRegression as MLR
from os import get_terminal_size


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

        def check_positive(value):
            try:
                ivalue = int(value)
                if ivalue <= 0:
                    raise argparse.ArgumentTypeError(
                        f"{value} is an invalid positive int value")
                return ivalue
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"{value} is an invalid positive int value")
            except Exception as e:
                print("Error parsing arguments: ", e)
                exit()

        parser.add_argument(
            '-i',
            '--nb-iterations',
            dest='iterations',
            type=check_positive,
            nargs='?',
            help='Number of iterations used for training.',
            default=20_000,
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
            help='Use the stochastic gradient descent on a batch' +
            ' for each iteration.',
            action='store_true',
            default=False
        )

        parser.add_argument(
            '--multi-mini-batch',
            '-mmb',
            help='Use the gradient descent on a mini-batch' +
            ' several times for each iterations.',
            action='store_true',
            default=False
        )

        args = parser.parse_args()

        if (args.stochastic, args.mini_batch,
            args.batch, args.multi_stochastic,
                args.multi_mini_batch).count(True) > 1:
            print("Error: you can only use one optimization method.")
            exit()

        if args.ratio is None:
            args.ratio = 0.5
        elif args.ratio < 0.0 or args.ratio > 1.0:
            print("Error: ratio must be between 0 and 1.")
            exit()

        if args.learning_rate <= 0.0 or args.learning_rate > 1.0:
            print("Error: learning rate must be greater than 0.")
            exit()

        return (
            args.dataset_path,
            args.loss_evolution,
            args.ratio,
            True if args.ratio != 0.0 else False,
            args.stochastic,
            args.mini_batch,
            args.multi_stochastic,
            args.multi_mini_batch,
            True if (not args.stochastic
                     and not args.mini_batch
                     and not args.multi_stochastic
                     and not args.multi_mini_batch) else False,
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
            raise Exception("The dataset is empty.")
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

        # Impute missing values
        y = dataset[target]
        x = dataset[features].to_numpy()
        mlr = MLR()
        full_x = mlr.knn_imputer(x, nb_neighbors=5)

        if y.isnull().values.any():
            raise Exception("The target contains missing values.")

        # Train/test split
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

    except Exception as e:
        print("Error :", e)
        exit(1)


def print_intro():

    options = {
        "Training mode": 'Batch' if batch
        else 'Mini-batch' if mini_batch
        else 'Stochastic' if stochastic
        else 'Multi-stochastic' if multi_stochastic
        else 'Multi-mini-batch',
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


def filter_house(y_train, house):
    """
    During training, filter the target to train one model for each house.
    If the training house is the same as the current house,
    the target is set to 1, else it is set to 0.
    """
    try:
        if house not in ("Ravenclaw", "Hufflepuff", "Slytherin", "Gryffindor"):
            return None
        ret = np.where(y_train == house, 1, 0)
        return ret.reshape(-1, 1)
    except Exception as e:
        print("Error filtering house: ", e)
        exit()


def print_model_stats():
    try:
        if test:
            y_hat = mlr.predict_(x_test)
            y_stat = filter_house(y_test, house)
        else:
            y_hat = mlr.predict_(x_norm)
            y_stat = filtered_y_train
        y_hat = np.array(
            [1 if x > 0.5 else 0 for x in y_hat]
        ).reshape(-1, 1)
        mlr.one_vs_all_stats(y_stat, y_hat)
    except Exception as e:
        print("Error printing model stats: ", e)


def predict(x_test, y_test, model, mlr, test_mode):
    try:
        print(f"\nTesting models on {test_mode} set.\n")
        prediction = np.empty((x_test.shape[0], 0))
        for house in houses:
            mlr.theta = model[house]
            y_hat = mlr.predict_(x_test)
            prediction = np.concatenate((prediction, y_hat), axis=1)
        y_hat = np.argmax(prediction, axis=1)
        y_hat = np.array([houses[i] for i in y_hat])
        y_test = y_test.to_numpy().reshape(-1)
        return y_hat, y_test
    except Exception:
        exit()


if __name__ == "__main__":

    try:

        # Get the options from the arguments parser
        (dataset_path,
         display_loss_evolution,
         split_ratio,
         test,
         stochastic,
         mini_batch,
         multi_stochastic,
         multi_mini_batch,
         batch,
         learning_rate,
         iterations) = parse_arguments()

        # Read the dataset
        dataset = read_dataset(dataset_path)

        # Set the features and target
        # Features are the courses, target is the house
        features = [
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

        # Set the training mode
        # Default is batch gradient descent, train on the whole dataset
        # Mini-batch gradient descent, train on a subset of the dataset
        # Stochastic gradient descent, train on one example at a time
        # Multi-stochastic is a mix of stochastic and mini-batch
        option = (
            batch,
            mini_batch,
            stochastic,
            multi_stochastic,
            multi_mini_batch
        ).index(True)

        # Split the dataset into training and test sets
        # Training set is used to train the models
        # Test set is used to test the models
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
        theta_shape = (x_test.shape[1] + 1, 1)

        # Normalize the dataset between 0 and 1,
        x_norm, x_min, x_max = MLR.normalize_train(x_train)
        if test:
            x_test = MLR.normalize_test(x_test, x_min, x_max)

        # Create the model, it will be saved in a .yml file
        model = {}
        model["x_min"] = x_min.reshape(1, -1)
        model["x_max"] = x_max.reshape(1, -1)
        model["features"] = features

        print_intro()

        # Train one model for each house :
        # Each model is trained on the same features,
        # but the target is filtered to train one model for each house
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

            # Set the training method depending on the option
            fit = (
                mlr.fit_,
                mlr.fit_mini_batch_,
                mlr.fit_stochastic_,
                mlr.fit_multi_stochastic_,
                mlr.fit_multi_mini_batch_
            )

            # Filter the target to train one model for each house
            filtered_y_train = filter_house(y_train, house)

            # Train the model
            fit[option](
                x_norm,
                filtered_y_train,
                display_loss_evolution
            )

            # Compute and print metrics for the model
            print_model_stats()

            if display_loss_evolution:
                mlr.plot_loss_evolution()

            # Save the model in a dictionary
            model[house] = mlr.theta

        # End of the training for loop

        if not test:
            # Save the model in a .yml file
            with open("models.yml", "w") as file:
                yaml.dump(model, file)
                print("Models saved in 'models.yml' file.")
            x_test = x_norm
            y_test = y_train
            test_mode = "train"
        else:
            print("Warning: test mode, the model won't be saved.\n")
            test_mode = "test"

        # Predict the houses for the test/training set depending on the mode
        y_hat, y_test = predict(x_test, y_test, model, mlr, test_mode)

        # Print the confusion matrix and the accuracy
        mlr.confusion_matrix_(
            y_test,
            y_hat,
            labels=houses,
            df_option=True,
            display=True
        )
        accuracy = mlr.accuracy_score_(y_hat, y_test) * 100
        print(f"\nAccuracy on {test_mode} set: {accuracy:.2f} %")

    except Exception as e:
        print("Error: ", e)
        exit()
