import argparse
import pandas
import seaborn as sn
import math
import numpy as np
import matplotlib.pyplot as plt


def parse_arguments() -> tuple:
    """
    Parse arguments of the program.
    """
    try:
        parser = argparse.ArgumentParser(
            prog="scatter_plot",
            description="This plot helps us to answer this question : " +
                        "What are the two features that are similar ?"
        )
        parser.add_argument(
            '-p', '--path',
            dest="dataset_path",
            type=str,
            default="../datasets/dataset_train.csv",
            help="Path to the dataset."
        )
        parser.add_argument(
            '-s', '--save-png',
            dest="save_png",
            action='store_true',
            help="Save the histogram in a png file."
        )
        args = parser.parse_args()
        return (
            args.dataset_path,
            args.save_png
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
        print("Error selecting columns: ", e)
        exit()

def mean(li):
    if (len(li) == 0):
        return None
    ret = 0
    for x in li:
        ret = ret + x
    return (float(ret / len(li)))

def var(li):
    ret = 0
    moy = mean(li)
    for x in li:
        ret = ret + ((x - moy) * (x - moy))
    return ret

def cov(arra1, arra2):
    M1 = mean(arra1.dropna().to_numpy())
    M2 = mean(arra2.dropna().to_numpy())
    deno = math.sqrt(var(arra1.dropna().to_numpy()) * var(arra2.dropna().to_numpy()))
    res = 0
    cout = 0
    arra1 = arra1.to_numpy()
    arra2 = arra2.to_numpy()
    for i in range(max(len(arra1), len(arra2))):
        if np.isnan(arra1[i]) or np.isnan(arra2[i]):
            continue
        res = res + ((arra1[i] - M1) * (arra2[i] - M2))
        count = cout + 1
    res = res / deno
    if res > 1 or res < -1:
        return 0
    return res

if __name__ == "__main__":

    dataset_path, save_png = parse_arguments()
    entire_dataset = read_dataset(dataset_path)
    dataset, feature_names = select_columns(entire_dataset)

    name = np.array([])
    val = np.array([])
    index = 0
    for x in feature_names:
        for y in feature_names:
            if x != y and x > y:
                cov_v = cov(dataset[x], dataset[y])
                print(index, " Correlation(", x.strip(), ",", y.strip(), ")",  cov_v)
                name = np.append(name, f"{x.strip()} vs. {y.strip()}")
                val = np.append(val, cov_v)
                index = index + 1

    # name = name.astype(int)

    dataplot = pandas.DataFrame(dict(Features=name, Correlation=val))
    plt.figure(figsize=(20, 10))
    sn.barplot(x='Features', y='Correlation', data=dataplot, errorbar=None)
    plt.xlabel("Features")
    plt.ylabel("Correlation")
    plt.title("Barplot of the correlation between features")
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    if save_png:
        plt.savefig("bar_plot_2.png")
    plt.show()

    plt.figure(figsize=(20, 10))
    sn.scatterplot(data=entire_dataset, x="Astronomy", y="Defense Against the Dark Arts", hue="Hogwarts House")
    plt.title("Scatter plot of the Defense Against the Dark Arts score depending on the Astronomy score")
    plt.tight_layout()
    if save_png:
        plt.savefig("scatter_plot.png")
    plt.show()
