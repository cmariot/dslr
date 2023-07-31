import numpy as np
import argparse
import pandas
import yaml
from my_logistic_regression import MyLogisticRegression
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
    x_min = np.amin(x_train, axis=0)
    x_max = np.amax(x_train, axis=0)
    x_norm = (x_train - x_min) / (x_max - x_min)
    return (
        x_norm,
        x_min.reshape(-1, 1),
        x_max.reshape(-1, 1)
    )


def normalize_test(x_test, x_min, x_max):
    x_norm = (x_test - x_min) / (x_max - x_min)
    return x_norm,


def denormalize_theta_minmax(theta, min_vals, max_vals):
    theta_denorm = np.zeros(theta.shape)
    theta_denorm[0] = theta[0] - np.sum(
        theta[1:] * min_vals / (max_vals - min_vals),
        dtype=np.float64
    )
    theta_denorm[1:] = theta[1:] / (max_vals - min_vals)
    return theta_denorm


if __name__ == "__main__":

    dataset_path, display_loss_evolution = parse_arguments()

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

    x_train_poly = MyLogisticRegression.add_polynomial_features(
        x_train,
        polynomial_degree
    )

    x_train_norm, x_min, x_max = normalize_train(x_train_poly)

    model = {}
    model["x_min"] = x_min
    model["x_max"] = x_max
    model["polynomial_degree"] = polynomial_degree

    theta_shape = (x_train_norm.shape[1] + 1, 1)

    for i, house in enumerate(houses):

        print(f"Training model {i + 1}/4 for house {house}")

        mlr = MyLogisticRegression(
            theta=np.zeros(theta_shape),
            max_iter=5_000,
            alpha=0.5,
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

        # # TESTS DENORMALIZE THETA
        # # Predict with normalized data
        # y_hat_norm = mlr.predict_(x_train_norm)
        # # Predict with denormalized data
        # mlr.theta = denormalize_theta_minmax(mlr.theta, x_min, x_max)
        # y_hat_denorm = mlr.predict_(x_train_poly)
        # # Compare the two predictions
        # print("Comparing the two predictions:")
        # print(np.array_equal(y_hat_norm, y_hat_denorm))  # -> False
        # # Concatenate the two predictions
        # print("Concatenating the two predictions:")
        # concat = np.concatenate((y_hat_norm, y_hat_denorm), axis=1)
        # print(concat)
        # # Compare the two predictions with pandas dataframe describe
        # print("Comparing the two predictions with pandas describe:")
        # print(pandas.DataFrame(concat).describe())  # -> Same
        # for idx in range(concat.shape[0]):
        #     print(concat[idx][0] - concat[idx][1])
        # # Les predictions ne sont pas tout a fait les memes,
        # # mais elles sont tres proches. (difference de l'ordre de 10e-17)
        # # En changeant le type de theta et x_min/max en np.float128, on
        # # obtient des predictions encore plus proches (ordre 10e-20),
        # # mais c'est plus long a entrainer.
        # # La difference vient probablement plus d'erreurs d'arrondis que
        # # d'une erreur dans le calcul de la prediction.
        # # On utilisera tout de meme x_min/max pour normaliser le set de test
        # # pour le rendu

    with open("models.yml", "w") as file:
        yaml.dump(model, file)
        print("Models saved in 'models.yml' file.\n")

    # Pas besoin de faire de prediction dans ce script,
    # ca sera fait dans logreg_predict.py.
    # Sauvegarde de ce qu'on avait fait vendredi :

    # polynomial test
    # normalize test

    # prediction = np.empty((x_train_norm.shape[0], 0))
    # for house in houses:
    #     mlr.theta = model[house]
    #     y_hat = mlr.predict_(x_train_norm)
    #     prediction = np.concatenate((prediction, y_hat), axis=1)

    # # Argmax sur les predictions pour trouver la maison la plus probable
    # # pour chaque ligne du dataset d'entrainement
    # y_hat = np.argmax(prediction, axis=1)
    # # On remplace les indices par les noms des maisons
    # y_hat = np.array([houses[i] for i in y_hat])
    # # On compare les predictions avec les vraies valeurs
    # y_train = y_train.to_numpy().reshape(-1)

    # # Confusion matrix
    # mlr.confusion_matrix_(
    #     y_train,
    #     y_hat,
    #     labels=houses,
    #     df_option=True,
    #     display=True
    # )

    # print("\nAccuracy on training set:")
    # accuracy = mlr.accuracy_score_(y_hat, y_train)
    # print(accuracy)
