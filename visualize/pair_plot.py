import seaborn as sns
import pandas
import argparse
import matplotlib.pyplot as plt


def parse_arguments() -> tuple:
    """
    Parse arguments of the program.
    """
    try:
        parser = argparse.ArgumentParser(
            prog="pair_plot",
            description=""
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
    """
    """
    try:
        without_index = dataset.drop("Index", axis='columns')
        return without_index
    except KeyError:
        return dataset
    except Exception as e:
        print("Error selecting columns: ", e)
        exit()


if __name__ == "__main__":

    dataset_path = parse_arguments()
    entire_dataset = read_dataset(dataset_path)
    dataset = select_columns(entire_dataset)

    # Check if there is a Hogwarts House column
    target = dataset["Hogwarts House"]
    target_name = target.name
    unique_targets = target.unique()

    palette = {
        "Gryffindor": "#BC2F2F",
        "Hufflepuff": "#8FBD61",
        "Ravenclaw": "#3D56D3",
        "Slytherin": "#E7E058"
    }

    sns.set(font_scale=0.5)

    pair_plot = sns.pairplot(
        data=dataset,
        hue=target_name,
        hue_order=unique_targets,
        palette=palette,
        dropna=True,
        kind="scatter",
        diag_kind="hist",
        markers=".",
        height=1,  # inch
        aspect=2,  # width = height * aspect
        corner=True,
        plot_kws={
            "alpha": 0.5,
        },
        diag_kws={
            "alpha": 0.5,
        }
    )

    pair_plot.fig.suptitle("Pair plot of numerical features")

    # Save the figure in a file, more readable than the figure
    plt.savefig("pair_plot.png", dpi=400)

    plt.show()
