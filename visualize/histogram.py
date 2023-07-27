import argparse
import pandas
import seaborn as sn
import math
import matplotlib.pyplot as plt


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

        numerical_dataset = dataset.select_dtypes(include="number")
        without_index = numerical_dataset.drop("Index", axis='columns')
        columns = without_index.columns
        return (
            without_index,
            columns
        )
    except KeyError:
        return numerical_dataset, numerical_dataset.columns
    except Exception as e:
        print(e)
        print("Error selecting columns: ", e)
        exit()

def mean(li):
    if (len(li) == 0):
        return None

    ret = 0
    for x in li:
        ret = ret + x
    return (float(ret / len(li)))

def std(li):
    lng = len(li)
    if (lng == 0):
        return None
    mn = mean(li)

    ret = 0
    for x in li:
        ret = ret + (x - mn)**2
    return (float(math.sqrt(ret / (lng - 1))))

if __name__ == "__main__":

    dataset_path = parse_arguments()
    entire_dataset = read_dataset(dataset_path)

    dataset, feature_names = select_columns(entire_dataset)

    label = dataset.columns

    histoinf = pandas.DataFrame(
        index=["std"],
        columns=dataset.columns,
        dtype = float
    )

    histogram = pandas.DataFrame(
        index=["Index"],
        columns=["Feature", "Std"] 
    )

    print("Standard deviation of all numerical features :")
    pandas.set_option('display.max_columns', None)
    for x in dataset.columns:
        std_val = std(dataset[x].dropna().to_numpy())
        histoinf.loc["std", x] = std_val
        elem = pandas.DataFrame([{"Feature" : x, "Std" : math.log(std_val)}])
        histogram = pandas.concat([histogram, elem], axis = 0, ignore_index=True)

    print(histoinf)
    print(histogram)
    sn.barplot(data=histogram, x="Feature", y="Std")
    plt.show()
    sn.histplot(data=entire_dataset, x="Care of Magical Creatures", hue="Hogwarts House")
    plt.show()


    #datahs = datahs.drop("Index", axis='columns')