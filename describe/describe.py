import argparse
import pandas
from utils import TinyStatistician

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
    try:
        print("all_col:", dataset.columns)
        print("all_col_type:", dataset.dtypes)
        print("all_col_shape:", dataset.shape)

        numerical_dataset = dataset.select_dtypes(include="number")
        columns = numerical_dataset.columns

        print("numerical_col:", numerical_dataset.columns)
        print("numerical_col_type:", numerical_dataset.dtypes)
        print("numerical_col_shape:", numerical_dataset.shape)

        return (
            numerical_dataset,
            columns
        )
    except Exception as e:
        print("Error selecting columns: ", e)
        exit()


def test(x):
    return 42


if __name__ == "__main__":

    dataset_path = parse_arguments()
    entire_dataset = read_dataset(dataset_path)
    dataset, feature_names = select_columns(entire_dataset)
    tstat = TinyStatistician()

    metrics = {
        "count": test,
        "mean": tstat.mean,
        "std": test,
        "min": test,
        "25%": test,
        "50%": test,
        "75%": test,
        "max": test 
    }

    description = pandas.DataFrame(
        index=metrics.keys(),
        columns=feature_names,
        dtype=float,
    )
    
    for feature in feature_names:
        for metric, function in metrics.items():
            description.loc[metric, feature] = function(feature)


    # Expected output:
    print(entire_dataset.describe())

    # Actual output:
    print(description)