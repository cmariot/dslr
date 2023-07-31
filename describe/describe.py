from argparse import ArgumentParser
import pandas
from metrics import TinyStatistician as Metrics
from os import get_terminal_size


def parse_arguments() -> tuple:
    """
    Parse the command line argument.
    Positional argument:
    - The program takes one positional argument, the path of the dataset.
    Optional arguments:
    - Bonus : [--bonus | -b] display more metrics.
    - Compare : [--compare | -c] compare with real describe().
    - Help : [--help | -h] display an help message.
    Usage:
      python describe.py [-b | --bonus] [-c | --compare] [-h | --help] data.csv
    """
    try:
        parser = ArgumentParser(
            prog="describe",
            description="This program takes a dataset path as argument. " +
            "It displays informations for all numerical features."
        )
        parser.add_argument(
            dest="dataset_path",
            type=str,
            help="Path to the dataset."
        )
        parser.add_argument(
            '-b', '--bonus',
            dest="bonus",
            action='store_true',
            help="Display more metrics."
        )
        parser.add_argument(
            '-c', '--compare',
            dest="compare",
            action='store_true',
            help="Comparaison with pandas.descibe()"
        )
        args = parser.parse_args()
        return (
            args.dataset_path,
            args.bonus,
            args.compare
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


def describe(dataset_path: str, bonus: bool = True, compare: bool = False):
    """
    Describe display numerical features metrics.
    Arguments:
    - dataset_path: path to the dataset.
    - bonus: display more metrics.
    - compare: compare with pandas.describe().
    """

    try:

        entire_dataset = read_dataset(dataset_path)
        dataset, feature_names = select_columns(entire_dataset)

        metrics = {
            "count": Metrics.count,
            "mean": Metrics.mean,
            "mode": Metrics.mode,
            "var": Metrics.var,
            "std": Metrics.std,
            "min": Metrics.min,
            "25%": Metrics.perc25,
            "50%": Metrics.perc50,
            "75%": Metrics.perc75,
            "max": Metrics.max,
            "range": Metrics.range,
            "iqr": Metrics.iqr,
            "aad": Metrics.aad,
            "cv": Metrics.cv,
        }

        if not bonus:
            del metrics["mode"]
            del metrics["var"]
            del metrics["range"]
            del metrics["iqr"]
            del metrics["aad"]
            del metrics["cv"]

        description = pandas.DataFrame(
            index=metrics.keys(),
            columns=feature_names,
            dtype=float,
        )

        for feature in feature_names:
            np_feature = dataset[feature].dropna().to_numpy()
            for metric, function in metrics.items():
                description.loc[metric, feature] = function(np_feature)

        with pandas.option_context(
            'display.max_columns', None,
            'display.width', get_terminal_size().columns
        ):

            print(description)
            if compare:
                if not bonus:
                    expected = dataset.describe()
                    print(expected, "\n")
                    print("OK") if description.equals(expected) \
                        else print("KO")
                else:
                    print("There are more metrics in this function" +
                          " than the pandas.descibe().")

        return description

    except Exception as error:
        print("Error: ", error)
        return None


if __name__ == "__main__":
    dataset_path, bonus, compare = parse_arguments()
    describe(dataset_path, bonus, compare)
