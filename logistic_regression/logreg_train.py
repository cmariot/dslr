import numpy as np
import argparse
import pandas
from sklearn.impute import KNNImputer
import yaml
from my_logistic_regression import MyLogisticRegression as MLR
# from logic_reg import MyLogisticRegression
# import seaborn as sns
# import matplotlib.pyplot as plt


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

        if (args.stochastic, args.mini_batch).count(True) > 1:
            print("Error: you can only use one optimization method.")
            exit()

        return (
            args.dataset_path,
            args.loss_evolution,
            args.test,
            args.stochastic,
            args.mini_batch,
            True
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
        stochastic,
        mini_batch,
        batch
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

    imputer = KNNImputer(n_neighbors=4)

    x_train = imputer.fit_transform(x_train)
    x_norm, x_min, x_max = MLR.normalize_train(x_train)

    if test_model:
        x_test = imputer.fit_transform(x_test)
        x_test = MLR.normalize_test(x_test, x_min, x_max)

    model = {}
    model["x_min"] = x_min.reshape(1, -1)
    model["x_max"] = x_max.reshape(1, -1)
    model["features"] = features

    theta_shape = (x_norm.shape[1] + 1, 1)

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
            mlr.fit_stochastic_,
            mlr.fit_mini_batch_,
            mlr.fit_
        )

        filtered_y_train = filter_house(y_train, house)

        fit[option](
            x_norm,
            filtered_y_train,
            display_loss_evolution
        )

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
