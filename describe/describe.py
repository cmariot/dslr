import argparse
import pandas
#import numpy as np
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

def countf(array):
    ret = 0
    for x in array:
        ret = ret + 1
    return ret

def minim(array):
    mini = array[0]
    for x in array:
        if (x < mini):
            mini = x
    return mini

def maxim(array):
    maxi = array[0]
    for x in array:
        if (x > maxi):
            maxi = x
    return maxi

def perc25(array):
    ts = TinyStatistician()
    return ts.percentile(array, 25)

def perc50(array):
    ts = TinyStatistician()
    return ts.percentile(array, 50)

def perc75(array):
    ts = TinyStatistician()
    return ts.percentile(array, 75)
 
if __name__ == "__main__":

    dataset_path = parse_arguments()
    entire_dataset = read_dataset(dataset_path)
    dataset, feature_names = select_columns(entire_dataset)
    tstat = TinyStatistician()

    metrics = {
        "count": countf,
        "mean": tstat.mean,
        "std": tstat.std,
        "min": minim,
        "25%": perc25,
        "50%": perc50,
        "75%": perc75,
        "max": maxim, 
    }

    datads = dataset.drop("Index", axis='columns')
    #datads.dropna(inplace=True)
    #datads.replace(np.nan, 0, inplace=True)

    featureds = datads.columns

    description = pandas.DataFrame(
        index=metrics.keys(),
        columns=featureds,
        dtype=float,
    )

    for feature in featureds:
        for metric, function in metrics.items():
            description.loc[metric, feature] = function(datads[feature].dropna().to_numpy())

    pandas.set_option('display.max_columns', None)
    #pandas.set_option("display.precision", 2)
    # Expected output:
    #print(datads.describe())

    # Actual output:
    print(description)