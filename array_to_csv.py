import numpy as np
import pandas


if __name__ == "__main__":

    arr = np.array(
        [
            ['0', '1', '2']
        ]
    )

    columns = []
    for col in range(arr.shape[1]):
        columns.append(f"Col{col}")

    dataframe = pandas.DataFrame(
        data=arr,
        columns=columns
    )

    print(dataframe)

    dataframe.to_csv("test.csv")
