import seaborn as sns
import pandas
import argparse
from matplotlib.pyplot import (savefig, show)


def parse_arguments() -> tuple:
    """
    Parse arguments of the program.
    """
    try:
        parser = argparse.ArgumentParser(
            prog="pair_plot",
            description="This plot helps us to choose features" +
                        " for logistic_regression."
        )
        parser.add_argument(
            '-p', '--path',
            dest="dataset_path",
            type=str,
            default="../datasets/dataset_train.csv",
            help="Path to the dataset."
        )
        parser.add_argument(
            '-t', '--target',
            dest="target_name",
            type=str,
            default="Hogwarts House",
            help="Target name."
        )
        parser.add_argument(
            '-c', '--corner',
            dest="corner",
            action='store_true',
            help="Plot only one part of the pair plot."
        )
        parser.add_argument(
            '-s', '--save-png',
            dest="save_png",
            action='store_true',
            help="Save the pair_plot in a png file."
        )
        args = parser.parse_args()
        return (
            args.dataset_path,
            args.target_name,
            args.corner,
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


def drop_index(dataset: pandas.DataFrame) -> pandas.DataFrame:
    """
    Remove the 'Index' column of the dataset.
    """
    try:
        # Remove empty columns
        dataset = dataset.dropna(axis=1, how='all')
        # Remove 'Index' column
        without_index = dataset.drop("Index", axis='columns')
        return without_index
    except KeyError:
        return dataset
    except Exception as e:
        print("Error selecting columns: ", e)
        exit()


def check_target(target_name, dataset) -> tuple:
    """
    Check if there is a 'target_name' column in the dataset.
    """
    try:
        target = dataset[target_name]
        # Check if objects ?
        unique_targets = target.unique()
        if len(unique_targets) == 4:
            # Hogwarts houses colors
            target_palette = {
                unique_targets[0]: "#BC2F2F",  # Gryffindor
                unique_targets[1]: "#8FBD61",  # Hufflepuff
                unique_targets[2]: "#3D56D3",  # Ravenclaw
                unique_targets[3]: "#E7E058"   # Slytherin
            }
        target_palette = None
        return (unique_targets, target_palette)
    except KeyError:
        print(f"Error: Target {target_name} not found in the dataset.")
        print("Consider using the --target option to specify a new target.")
        exit()
    except Exception as error:
        print("Error:", error)
        exit(1)


if __name__ == "__main__":

    (
        dataset_path,
        target_name,
        corner,
        save_png
    ) = parse_arguments()

    entire_dataset = read_dataset(dataset_path)
    dataset = drop_index(entire_dataset)
    (unique_targets, target_palette) = check_target(target_name, dataset)

    sns.set(font_scale=0.5)

    pair_plot = sns.pairplot(
        data=dataset,
        hue=target_name,
        hue_order=unique_targets,
        palette=target_palette,
        dropna=True,
        kind="scatter",
        diag_kind="hist",
        markers=".",
        height=1,  # inch
        aspect=2,  # width = height * aspect
        corner=corner,
        plot_kws={
            "alpha": 0.5,
        },
        diag_kws={
            "alpha": 0.5,
        }
    )

    # Figure title
    pair_plot.fig.suptitle(
        f"Pair plot representing feature correlations of {target_name}." +
        " (Scatter plot matrix with histograms)",
        y=0.9975
    )

    # Save the figure in a file, more readable than the figure
    if save_png:
        savefig("pair_plot.png", dpi=400)

    show()
