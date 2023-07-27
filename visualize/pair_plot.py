import seaborn as sns
import pandas
import argparse
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

        numerical_dataset = dataset.select_dtypes(include=("number", "object"))
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


if __name__ == "__main__":

    dataset_path = parse_arguments()
    entire_dataset = read_dataset(dataset_path)
    dataset, columns = select_columns(entire_dataset)


    palette = {
        "Gryffindor": "#740001",
        "Hufflepuff": "#ecb939",
        "Ravenclaw": "#0e1a40",
        "Slytherin": "#1a472a"
    }

    sns.set(font_scale=0.5)
    sns.pairplot(
        data=dataset,
        hue='Hogwarts House',
        hue_order=["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"],
        palette=palette,
        dropna=True,
        kind="scatter",
        diag_kind="hist",
        markers=".",
        height=1,  # inch
        aspect=2,  # width / height
        plot_kws={
        },
        diag_kws={
        }
    )
    plt.show()