import pandas
import sklearn.metrics as skm


if __name__ == "__main__":

    truth = pandas.read_csv("../datasets/truth.csv")
    prediction = pandas.read_csv("./houses.csv")

    y_true = truth["Hogwarts House"].to_numpy()
    y_pred = prediction["Hogwarts House"].to_numpy()

    # Accuracy
    accuracy = skm.accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}")

