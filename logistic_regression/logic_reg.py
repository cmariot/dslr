import numpy as np
import math as mat
from tqdm import tqdm
import matplotlib.pyplot as plt
class MyLogisticRegression():
    """
        Description:
        My personnal logistic regression to classify things.
    """
    def __init__(self, theta, alpha=0.001, max_iter=1000):
        if not MyLogisticRegression.check_matix(theta):
            print("+1")
            return None
        if theta.shape[1] != 1:
            print("+2")
            return None
        if not type(alpha) is float:
            return None
        if not type(max_iter) is int:
            return None

        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta
        self.losses = []
        self.r2_scores = []

    @staticmethod
    def check_matix(mat):
        if not (isinstance(mat, np.ndarray)):
            print("mat1")
            return False
        if mat.dtype != "int64" and mat.dtype != "float64":
            print("mat2")
            return False
        if len(mat.shape) != 2:
            print("mat3")
            return False
        if (mat.size == 0):
            print("mat4")
            return False
        return True

    @staticmethod
    def sigmoid(x):
        sig = lambda x : 1/ (1 + mat.exp(-x))
        return (np.array([[sig(elem)] for elem in x]))

    @staticmethod
    def grad(x, y, theta):
        X_prime = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1).astype(float)
        return (np.matmul(X_prime.T, (MyLogisticRegression.sigmoid(np.matmul(X_prime, theta)) - y)) / x.shape[0])

    def predict_(self, x):
        if (not MyLogisticRegression.check_matix(x)):
            return (None)
        if x.shape[1] != self.theta.shape[0] - 1:
            return None
        X_prime = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1).astype(float)
        return (MyLogisticRegression.sigmoid(np.matmul(X_prime, self.theta)))

    def loss_elem_(self, y, yhat):
        if (not MyLogisticRegression.check_matix(y) or not MyLogisticRegression.check_matix(yhat)):
            return None
        if not y.shape[1] == 1 or not yhat.shape[1] == 1:
            return None
        if not y.shape[0] == yhat.shape[0]:
            return None
        ret = np.zeros((y.shape[0], 1))

        for idx in range(y.shape[0]):
            ret[idx] =  y[idx][0] * mat.log(yhat[idx][0] + 1e-15) + (1 - y[idx][0])* mat.log(1 - yhat[idx][0] + 1e-15)
        return ret

    def loss_(self, y, yhat):
        if (not MyLogisticRegression.check_matix(y) or not MyLogisticRegression.check_matix(yhat)):
            return None
        if y.shape[1] != 1 or yhat.shape[1] != 1:
            return None
        if not y.shape[0] == yhat.shape[0]:
            return None

        ret = 0
        for idx in range(y.shape[0]):
            ret = ret + ( y[idx][0] * mat.log(yhat[idx][0] + 1e-15) + (1 - y[idx][0])* mat.log(1 - yhat[idx][0] + 1e-15))
        return (- ret / y.shape[0])

    def fit_(self, x, y):
        if not MyLogisticRegression.check_matix(x) or not MyLogisticRegression.check_matix(y):
            print("1")
            return None
        if y.shape[1] != 1 or x.shape[0] != y.shape[0]:
            print("2")
            return None
        if x.shape[1] != self.theta.shape[0] - 1:
            print("3")
            return None
        if (y.shape[0] != x.shape[0]):
            print("4")
            return None

        for it in tqdm(range(self.max_iter)):
            self.theta = self.theta - (self.alpha * MyLogisticRegression.grad(x, y, self.theta))
            if True in np.isnan(self.theta):
                return None
            y_hat = self.predict_(x)
            self.losses.append(
                self.loss_(y, y_hat)
            )
            self.r2_scores.append(
                self.r2_score(y, y_hat)
            )
        print()
        return self.theta

    def r2_score(self, y, y_hat):
        """
        Return the R2 score.
        """
        try:
            y_mean = np.mean(y)
            return 1 - (np.sum(np.square(y_hat - y)) /
                        np.sum(np.square(y - y_mean)))
        except Exception:
            return None

    def plot_loss_evolution(self):
        try:

            fig = plt.figure()

            # Plot the losses
            ax1 = fig.add_subplot(111)
            ax1.set_xlabel("Iteration")
            ax1.plot(self.losses, color="red", label="Loss")
            ax1.legend(["Loss"], loc="center right",
                       bbox_to_anchor=(0.95, 0.45))
            ax1.set_ylabel("Loss", color="red")
            for tl in ax1.get_yticklabels():
                tl.set_color("red")
                tl.set_fontsize(9)
            plt.text(0.6, 0.10, f"last loss = {self.losses[-1]:.2f}",
                     transform=plt.gca().transAxes,
                     fontsize=11, verticalalignment='top', color="red")

            # Plot the R2 scores
            ax2 = ax1.twinx()
            ax2.plot(self.r2_scores, color="blue", label="R2 score")
            ax2.legend(["R2 score"], loc="center right",
                       bbox_to_anchor=(0.95, 0.55))
            ax2.set_ylabel("R2 score", color="blue")
            for tl in ax2.get_yticklabels():
                tl.set_color("blue")
            plt.text(0.6, 0.925, f"last R2-score = {self.r2_scores[-1]:.2f}",
                     transform=plt.gca().transAxes,
                     fontsize=11, verticalalignment='top', color="blue")

            plt.title("Metrics evolution during training")
            plt.grid(linestyle=':', linewidth=0.5)
            plt.show()
        except Exception:
            return None
