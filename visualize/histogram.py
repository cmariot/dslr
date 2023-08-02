import argparse
import pandas
import seaborn as sn
import math
import matplotlib.pyplot as plt


def parse_arguments() -> tuple:
    """
    Parse arguments of the program.
    """
    try:
        parser = argparse.ArgumentParser(
            prog="histogram",
            description="This plot helps us to answer this question : " +
                        "Which Hogwarts course has a homogeneous score " +
                        "distribution between all four houses ?"
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

    dataset_path, save_png = parse_arguments()
    entire_dataset = read_dataset(dataset_path)

    dataset, feature_names = select_columns(entire_dataset)

    histoinf = pandas.DataFrame(
        index=["std"],
        columns=dataset.columns,
        dtype=float
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
        elem = pandas.DataFrame([{"Feature": x, "Std": math.log(std_val)}])
        histogram = pandas.concat([histogram, elem], axis=0, ignore_index=True)

    print(histoinf)
    print(histogram)

    plt.figure(figsize=(20, 10))
    sn.barplot(data=histogram, x="Feature", y="Std")
    plt.xticks(rotation=15)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.xlabel("Feature")
    plt.ylabel("Logarithm of the standard deviation")
    plt.title("Standard deviation of all numerical features")
    if save_png:
        plt.savefig("bar_plot_1.png", dpi=400)
    plt.show()

    most_homogeneous_feature = \
        histogram.loc[histogram["Std"].idxmin()]["Feature"]

    plt.figure(figsize=(20, 10))
    sn.histplot(
        data=entire_dataset,
        x=most_homogeneous_feature,
        hue="Hogwarts House"
    )
    plt.xlabel(f"{most_homogeneous_feature} scores")
    plt.ylabel("Number of students")
    plt.title(f"Most homogeneous feature : {most_homogeneous_feature}")
    if save_png:
        plt.savefig("histogram.png", dpi=400)
    plt.show()
