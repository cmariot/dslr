import numpy as np
from pandas import DataFrame
from time import time
from os import get_terminal_size
import matplotlib.pyplot as plt


class MyLogisticRegression:
    """
    Description: My logistic regression to classify things.
    """

    # We consider l2 penality only.
    # One may wants to implement other penalities
    supported_penalities = ['l2']

    def checkargs_init(func):
        def wrapper(self,
                    theta=np.zeros((2, 1)),
                    learning_rate=0.1,
                    max_iter=1_000,
                    penality=None,
                    lambda_=0.0,
                    stochastic=False,
                    mini_batch=False,
                    batch_size=32):
            try:
                if not isinstance(theta, np.ndarray):
                    raise TypeError("theta must be a numpy.ndarray")
                m, n = theta.shape
                if m == 0 or n != 1:
                    raise ValueError(
                        "theta must be a np.ndarray of shape (n, 1)")
                if not isinstance(learning_rate, (int, float)):
                    raise TypeError("learning_rate must be a float")
                if not isinstance(max_iter, int):
                    raise TypeError("max_iter must be an int")
                if not isinstance(penality, (str, type(None))):
                    raise TypeError("penality must be a string or None")
                if penality is not None and \
                        penality not in self.supported_penalities:
                    print("Warning : penality not supported")
                if not isinstance(lambda_, (int, float)):
                    raise TypeError("lambda_ must be a float")
                if lambda_ < 0:
                    raise ValueError("lambda_ must be positive")
                if not isinstance(stochastic, bool):
                    raise TypeError("stochastic must be a boolean")
                if not isinstance(mini_batch, bool):
                    raise TypeError("mini_batch must be a boolean")
                if not isinstance(batch_size, int):
                    raise TypeError("batch_size must be an int")
                if batch_size < 1:
                    raise ValueError("batch_size must be positive")
                if stochastic or mini_batch:
                    if penality is not None:
                        raise ValueError(
                            "penality not supported with this option")
                return func(self,
                            theta,
                            learning_rate,
                            max_iter,
                            penality,
                            lambda_,
                            stochastic,
                            mini_batch,
                            batch_size)
            except Exception as e:
                print("MyLogisticRegression init error :", e)
                return None
        return wrapper

    @checkargs_init
    def __init__(self, theta, learning_rate, max_iter,
                 penality, lambda_, stochastic,
                 mini_batch, batch_size):
        """
        Args:
            theta: has to be a numpy.ndarray, a vector of dimension (number of
                features + 1, 1).
            learning_rate: has to be a float, the learning rate
            max_iter: has to be an int, the number of iterations done during
                the gradient descent
            lambda_: has to be a float, the regularization parameter
            penality: has to be a string, either 'l2' or None
        """
        try:
            self.learning_rate = learning_rate
            self.max_iter = max_iter
            self.theta = theta
            self.penality = penality
            self.lambda_ = lambda_ if penality in self.supported_penalities \
                else 0.0
            self.losses = []
            self.f1_scores = []
            self.stochastic = stochastic
            self.mini_batch = mini_batch
            self.batch_size = batch_size
        except Exception as e:
            print("MyLogisticRegression init error :", e)
            return None

    def add_polynomial_features(x, power):
        """
        Add polynomial features to matrix x by raising its columns
        to every power in the range of 1 up to the power given in argument.
        Args:
            x: has to be an numpy.ndarray, a matrix of shape m * n.
            power: has to be an int, the power up to which the columns
                    of matrix x are going to be raised.
        Returns:
            The matrix of polynomial features as a numpy.ndarray,
                of shape m * (np), containg the polynomial feature values
                for all training examples.
            None if x is an empty numpy.ndarray.
        Raises:
            This function should not raise any Exception.
        """
        try:
            if not isinstance(x, np.ndarray):
                return None
            m, n = x.shape
            if m == 0 or n == 0:
                return None
            if not isinstance(power, int) or power < 1:
                return None
            polynomial_matrix = x.copy()
            for i in range(2, power + 1):
                new_column = x ** i
                polynomial_matrix = np.c_[polynomial_matrix, new_column]
            return polynomial_matrix
        except Exception:
            return None

    def normalize_train(x_train):
        try:
            x_min = np.amin(x_train, axis=0)
            x_max = np.amax(x_train, axis=0)
            x_norm = (x_train - x_min) / (x_max - x_min)
            return (
                x_norm,
                x_min.reshape(-1, 1),
                x_max.reshape(-1, 1)
            )
        except Exception as e:
            print("Error normalizing training set: ", e)
            return None, None, None

    def normalize_test(x_test, x_min, x_max):
        try:
            x_max = x_max.reshape(1, -1)
            x_min = x_min.reshape(1, -1)
            return (x_test - x_min) / (x_max - x_min)
        except Exception as e:
            print("Error normalizing test set: ", e)
            return None

    def checkargs_sigmoid_(func):
        def wrapper(self, x):
            try:
                if not isinstance(x, np.ndarray):
                    raise TypeError(
                        "x must be a numpy.ndarray")
                m = x.shape[0]
                if m == 0 or x.shape != (m, 1):
                    raise ValueError(
                        "x must be a numpy.ndarray of shape (m, 1)")
                return func(self, x)
            except Exception as e:
                print("MyLogisticRegression sigmoid_ error :", e)
                return None
        return wrapper

    @checkargs_sigmoid_
    def sigmoid_(self, x):
        try:
            return 1 / (1 + np.exp(-x))
        except Exception as e:
            print("MyLogisticRegression sigmoid_ error :", e)
            return None

    def checkargs_predict_(func):
        def wrapper(self, x):
            try:
                if not isinstance(x, np.ndarray):
                    print("x must be a numpy.ndarray\n\n")
                    return None
                m, n = x.shape
                if m == 0 or n == 0:
                    return None
                elif self.theta.shape != ((n + 1), 1):
                    return None
                return func(self, x)
            except Exception:
                return None
        return wrapper

    @checkargs_predict_
    def predict_(self, x):
        try:
            m = x.shape[0]
            x_prime = np.concatenate((np.ones((m, 1)), x), axis=1)
            y_hat = self.sigmoid_(x_prime.dot(self.theta))
            return y_hat
        except Exception as e:
            print(e)
            return None

    def checkargs_l2_(func):
        def wrapper(self):
            if not isinstance(self.theta, np.ndarray):
                return None
            elif self.theta.size == 0:
                return None
            return func(self)
        return wrapper

    @checkargs_l2_
    def l2(self):
        """
        Computes the L2 regularization of a non-empty numpy.ndarray,
        without any for-loop.
        Args:
            theta: has to be a numpy.ndarray, a vector of shape n * 1.
        Returns:
            The L2 regularization as a float.
            None if theta in an empty numpy.ndarray.
        Raises:
            This function should not raise any Exception.
        """
        try:
            theta_prime = np.copy(self.theta)
            theta_prime[0, 0] = 0.0
            regularization = np.dot(theta_prime.T, theta_prime)
            return float(regularization[0, 0])
        except Exception:
            return None

    def checkargs_reg_log_loss_(func):
        def wrapper(self, y, y_hat):
            try:
                if not isinstance(y, np.ndarray) \
                    or not isinstance(y_hat, np.ndarray) \
                        or not isinstance(self.theta, np.ndarray):
                    print("y, y_hat or theta is not a np.ndarray")
                    return None
                m = y.shape[0]
                n = self.theta.shape[0]
                if m == 0 or n == 0:
                    print("m or n is 0")
                    return None
                if y.shape != (m, 1) \
                    or y_hat.shape != (m, 1) \
                        or self.theta.shape != (n, 1):
                    print("y, y_hat or theta has a wrong shape")
                    return None
                return func(self, y, y_hat)
            except Exception as e:
                print(e)
                return None
        return wrapper

    def vec_log_loss_(self, y, y_hat, eps=1e-15):
        """
            Compute the logistic loss value.
            Args:
                y: has to be an numpy.ndarray, a vector of shape m * 1.
                y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
                eps: epsilon (default=1e-15)
            Returns:
                The logistic loss value as a float.
                None on any error.
            Raises:
                This function should not raise any Exception.
        """
        try:
            if not all(isinstance(x, np.ndarray) for x in [y, y_hat]):
                return None
            m, n = y.shape
            if m == 0 or n == 0:
                return None
            elif y_hat.shape != (m, n):
                return None
            y_hat_clipped = np.clip(y_hat, eps, 1 - eps)
            const = -1.0 / m
            dot1 = np.dot(y.T, np.log(y_hat_clipped))
            dot2 = np.dot((1 - y).T, np.log(1 - y_hat_clipped))
            return (const * (dot1 + dot2)[0, 0])
        except Exception:
            return None

    @checkargs_reg_log_loss_
    def loss_(self, y, y_hat):
        """
        Computes the regularized loss of a logistic regression model
        from two non-empty numpy.ndarray, without any for loop.
        Args:
            y: has to be an numpy.ndarray, a vector of shape m * 1.
            y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
            theta: has to be a numpy.ndarray, a vector of shape n * 1.
            lambda_: has to be a float.
        Returns:
            The regularized loss as a float.
            None if y, y_hat, or theta is empty numpy.ndarray.
            None if y and y_hat do not share the same shapes.
        Raises:
            This function should not raise any Exception.
        """
        try:
            loss = self.vec_log_loss_(y, y_hat)
            if loss is None:
                return None
            l2_ = self.l2()
            if l2_ is None:
                return loss
            reg = (self.lambda_ / (2 * y.shape[0])) * l2_
            return loss + reg
        except Exception:
            return None

    def checkargs_gradient_(func):
        def wrapper(self, x, y):
            try:
                if not isinstance(y, np.ndarray) \
                        or not isinstance(x, np.ndarray):
                    print("y or x is not a np.ndarray")
                    return None
                m, n = x.shape
                if m == 0 or n == 0:
                    print("m or n is 0")
                    return None
                if y.shape != (m, 1) \
                    or x.shape != (m, n) \
                        or self.theta.shape != (n + 1, 1):
                    print("y, x or theta has a wrong shape")
                    return None
                return func(self, x, y)
            except Exception:
                return None
        return wrapper

    @checkargs_gradient_
    def gradient_(self, x, y):
        try:
            m, _ = x.shape
            X_prime = np.hstack((np.ones((m, 1)), x))
            y_hat = self.predict_(x)
            if y_hat is None:
                return None
            theta_prime = self.theta.copy()
            theta_prime[0, 0] = 0.0
            return (X_prime.T.dot(y_hat - y) + (self.lambda_ * theta_prime)) \
                / m
        except Exception:
            return None

    def ft_progress(self, iterable,
                    length=get_terminal_size().columns - 4,
                    fill='█',
                    empty='░',
                    print_end='\r'):
        """
        Progress bar generator.
        """

        def get_elapsed_time_str(elapsed_time):
            """
            Return the elapsed time as str.
            """
            if elapsed_time < 60:
                return f'[Elapsed-time {elapsed_time:.2f} s]'
            elif elapsed_time < 3600:
                return f'[Elapsed-time {elapsed_time / 60:.0f} m]'
            else:
                return f'[Elapsed-time {elapsed_time / 3600:.0f} h]'

        def get_eta_str(eta):
            """
            Return the Estimed Time Arrival as str.
            """
            if eta == 0.0:
                return ' [DONE]                         '
            elif eta < 60:
                return f' [{eta:.0f} s remaining]       '
            elif eta < 3600:
                return f' [{eta / 60:.0f} m remaining]  '
            else:
                return f' [{eta / 3600:.0f} h remaining]'

        try:
            print()
            total = len(iterable)
            start = time()
            for i, item in enumerate(iterable, start=1):
                elapsed_time = time() - start
                et_str = get_elapsed_time_str(elapsed_time)
                eta_str = get_eta_str(elapsed_time * (total / i - 1))
                filled_length = int(length * i / total)
                percent_str = f'[{(i / total) * 100:6.2f} %] '
                progress_str = str(fill * filled_length
                                   + empty * (length - filled_length))
                counter_str = f'  [{i:>{len(str(total))}}/{total}] '
                bar = ("\033[F\033[K  " + progress_str + "\n"
                       + counter_str
                       + percent_str
                       + et_str
                       + eta_str)
                print(bar, end=print_end)
                yield item
            print()
        except Exception:
            print("Error: ft_progress")
            return None

    def checkargs_fit_(func):
        def wrapper(self, x_train, y_train, compute_metrics=False):
            try:
                if not isinstance(y_train, np.ndarray) \
                        or not isinstance(x_train, np.ndarray):
                    return None
                m, n = x_train.shape
                if m == 0 or n == 0:
                    return None
                if y_train.shape != (m, 1) \
                    or x_train.shape != (m, n) \
                        or self.theta.shape != (n + 1, 1):
                    return None
                if not isinstance(compute_metrics, bool):
                    return None
                return func(self, x_train, y_train, compute_metrics)
            except Exception as e:
                print(e)
                return None
        return wrapper

    @checkargs_fit_
    def fit_(self, x_train, y_train, compute_metrics):
        """
        Fits the model to the training dataset contained in x and y.
        """
        try:
            for _ in self.ft_progress(range(self.max_iter)):
                gradient = self.gradient_(x_train, y_train)
                if gradient is None:
                    return None
                self.theta -= (self.learning_rate * gradient)
                if compute_metrics:
                    y_hat = self.predict_(x_train)
                    self.losses.append(self.loss_(y_train, y_hat))
                    y_hat = np.where(y_hat >= 0.8, 1, 0)
                    self.f1_scores.append(self.f1_score_(y_train, y_hat))
            print()
            return self.theta
        except Exception:
            return None

    @checkargs_fit_
    def fit_stochastic_(self, x_train, y_train, compute_metrics):
        """
        Fits the model to the training dataset contained in x and y.
        This method uses the stochastic gradient descent.
        The gradient is computed on a random sample of the dataset.
        """
        try:
            for _ in self.ft_progress(range(self.max_iter)):
                idx = np.random.randint(0, x_train.shape[0])
                x = x_train[idx, :].reshape(1, -1)
                y = y_train[idx, :].reshape(1, -1)
                gradient = self.gradient_(x, y)
                if gradient is None:
                    return None
                self.theta -= (self.learning_rate * gradient)
                if compute_metrics:
                    y_hat = self.predict_(x_train)
                    self.losses.append(self.loss_(y_train, y_hat))
                    y_hat = np.where(y_hat >= 0.8, 1, 0)
                    self.f1_scores.append(self.f1_score_(y_train, y_hat))
            print()
            return self.theta
        except Exception:
            return None

    @checkargs_fit_
    def fit_mini_batch_(self, x_train, y_train, compute_metrics):
        """
        Fits the model to the training dataset contained in x and y.
        This method uses the stochastic gradient descent.
        The gradient is computed on a random sample of the dataset.
        """
        try:
            train_set = np.concatenate((x_train, y_train), axis=1)
            for _ in self.ft_progress(range(self.max_iter)):
                index_start = np.random.randint(0, x_train.shape[0])
                index_end = index_start + self.batch_size
                if index_end > x_train.shape[0]:
                    batch_first_part = \
                        train_set[index_start:, :]
                    batch_second_part = \
                        train_set[:index_end - x_train.shape[0], :]
                    batch = np.concatenate(
                        (batch_first_part, batch_second_part),
                        axis=0
                    )
                else:
                    batch = train_set[index_start:index_end, :]
                x = batch[:, :-1]
                y = batch[:, -1].reshape(-1, 1)
                gradient = self.gradient_(x, y)
                if gradient is None:
                    return None
                self.theta -= (self.learning_rate * gradient)
                if compute_metrics:
                    y_hat = self.predict_(x_train)
                    self.losses.append(self.loss_(y_train, y_hat))
                    y_hat = np.where(y_hat >= 0.8, 1, 0)
                    self.f1_scores.append(self.f1_score_(y_train, y_hat))
            print()
            return self.theta
        except Exception:
            return None

    @checkargs_fit_
    def fit_multi_mini_batch_(self, x_train, y_train, compute_metrics):
        """
        Fits the model to the training dataset contained in x and y.
        This method uses the stochastic gradient descent.
        The gradient is computed on a random sample of the dataset.
        """

        try:
            train_set = np.concatenate((x_train, y_train), axis=1)
            for _ in self.ft_progress(range(self.max_iter)):
                for idx in range(0, train_set.shape[0] - 32, 32):
                    x = x_train[idx:idx + 32, :]
                    y = y_train[idx:idx + 32, :]

                    gradient = self.gradient_(x, y)
                    if gradient is None:
                        return None
                    self.theta -= (self.learning_rate * gradient)
                    if compute_metrics:
                        y_hat = self.predict_(x_train)
                        self.losses.append(self.loss_(y_train, y_hat))
                        y_hat = np.where(y_hat >= 0.8, 1, 0)
                        self.f1_scores.append(self.f1_score_(y_train, y_hat))
            print()
            return self.theta
        except Exception:
            return None

    @checkargs_fit_
    def fit_multi_stochastic_(self, x_train, y_train, compute_metrics):
        """
        Fits the model to the training dataset contained in x and y.
        This method uses the stochastic gradient descent.
        The gradient is computed on a random sample of the dataset.
        """
        try:
            for i in self.ft_progress(range(self.max_iter)):
                list = np.arange(0, x_train.shape[0])
                for j in range(self.batch_size):
                    idx = np.random.randint(0, list.size)
                    x = x_train[idx, :].reshape(1, -1)
                    y = y_train[idx, :].reshape(1, -1)
                    list = np.delete(list, idx)
                    if list.size == 0:
                        list = np.arange(0, x_train.shape[0])
                    gradient = self.gradient_(x, y)
                    if gradient is None:
                        return None
                    self.theta -= (self.learning_rate * gradient)
                    if compute_metrics:
                        y_hat = self.predict_(x_train)
                        self.losses.append(self.loss_(y_train, y_hat))
                        y_hat = np.where(y_hat >= 0.8, 1, 0)
                        self.f1_scores.append(self.f1_score_(y_train, y_hat))
            print()
            return self.theta
        except Exception as e:
            print("Exception:", e)
            return None

# def fit_one_by_one_feature(self, x_train, y_train, compute_metrics):
#     try:
#         for _ in self.ft_progress(range(self.max_iter)):
#             m, n = x_train.shape
#             col = np.random.randint(n)
#             x_temp = x_train[:, col].reshape(-1, 1)
#             x_prime = np.concatenate((np.ones((m, 1)), x_temp), axis=1)
#             theta_tmp = np.array([self.theta[0], self.theta[col + 1]])
#             y_hat = self.sigmoid_(x_prime.dot(theta_tmp))
#             gradient = x_prime.T.dot(y_hat - y_train) / m
#             self.theta[col + 1] -= (self.learning_rate * gradient[1])
#             if compute_metrics:
#                 y_hat = self.predict_(x_train)
#                 self.losses.append(self.loss_(y_train, y_hat))
#                 y_hat = np.where(y_hat >= 0.5, 1, 0)
#                 self.f1_scores.append(self.f1_score_(y_train, y_hat))
#         print()
#         return self.theta
#     except Exception as e:
#         print(e)
#         return None

    def one_vs_all_stats(self, y, y_hat, pos_label=1):
        try:
            if not isinstance(y, np.ndarray) \
                    or not isinstance(y_hat, np.ndarray):
                return None
            if y.shape != y_hat.shape:
                return None
            if y.size == 0:
                return None
            if not isinstance(pos_label, (int, str)):
                return None
            st = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

            for idx in range(y.shape[0]):
                if (y_hat[idx] == y[idx]):
                    if y[idx] == pos_label:
                        st['tp'] += 1
                    else:
                        st['tn'] += 1
                else:
                    if y[idx] == pos_label:
                        st['fn'] += 1
                    else:
                        st['fp'] += 1

            nb = st['tp'] + st['tn'] + st['fp'] + st['fn']
            accuracy = (st['tp'] + st['tn']) / nb
            if (st['tp'] + st['fp']) != 0:
                prec = st['tp'] / (st['tp'] + st['fp'])
            else:
                prec = None
            if (st['tp'] + st['fn']) != 0:
                reca = st['tp'] / (st['tp'] + st['fn'])
            else:
                reca = None
            if prec is not None and reca is not None and prec + reca != 0:
                f1 = (2 * prec * reca) / (prec + reca)
            else:
                f1 = None

            stats = {
                'accuracy': accuracy,
                'true positive': st['tp'],
                'false positive': st['fp'],
                'true negative': st['tn'],
                'false negative': st['fn'],
                'total': nb,
                'f1_score': f1,
                'precision': prec,
                'recall': reca,
            }
            df = DataFrame(
                stats,
                index=[""])
            print(df, "\n\n")

            # print("Statistics : ", st['tp'] + st['tn'], "/", nb, "=",
            #       accuracy * 100, "%", " accuracy with :")
            # print("\t", st['fn'], " false negativ (missed)")
            # print("\t", st['fp'], "false positiv (errors)")
            # print("\t f1_score :", f1, " with : ")
            # print("\t\t precision score :", reca)
            # print("\t\t recall score :", prec)

        except Exception as e:
            print(e)
            return None

    def accuracy_score_(self, y, y_hat):
        """
        Compute the accuracy score.
        Accuracy tells you the percentage of predictions that are accurate
        (i.e. the correct class was predicted).
        Accuracy doesn't give information about either error type.
        Args:
            y: a numpy.ndarray for the correct labels
            y_hat:a numpy.ndarray for the predicted labels
        Returns:
            The accuracy score as a float.
            None on any error.
        Raises:
            This function should not raise any Exception.
        """

        try:
            if not isinstance(y, np.ndarray) \
                    or not isinstance(y_hat, np.ndarray):
                return None
            if y.shape != y_hat.shape:
                return None
            if y.size == 0:
                return None
            true = np.where(y == y_hat)[0].shape[0]
            return true / y.size

        except Exception:
            return None

    def precision_score_(self, y, y_hat, pos_label=1):
        """
        Compute the precision score.
        Precision tells you how much you can trust your
        model when it says that an object belongs to Class A.
        More precisely, it is the percentage of the objects
        assigned to Class A that really were A objects.
        You use precision when you want to control for False positives.
        Args:
            y: a numpy.ndarray for the correct labels
            y_hat: a numpy.ndarray for the predicted labels
            pos_label: str or int, the class on which to report
                    the precision_score (default=1)
        Return:
            The precision score as a float.
            None on any error.
        Raises:
            This function should not raise any Exception.
        """
        try:
            if not isinstance(y, np.ndarray) \
                    or not isinstance(y_hat, np.ndarray):
                return None
            if y.shape != y_hat.shape:
                return None
            if y.size == 0 or y_hat.size == 0:
                return None
            if not isinstance(pos_label, (int, str)):
                return None
            tp = np.sum(np.logical_and(y == pos_label, y == y_hat))
            fp = np.sum(np.logical_and(y != pos_label, y_hat == pos_label))
            predicted_positive = tp + fp
            if predicted_positive == 0:
                return None
            return tp / predicted_positive

        except Exception:
            return None

    def recall_score_(self, y, y_hat, pos_label=1):
        """
        Compute the recall score.
        Recall tells you how much you can trust that your
        model is able to recognize ALL Class A objects.
        It is the percentage of all A objects that were properly
        classified by the model as Class A.
        You use recall when you want to control for False negatives.
        Args:
            y:a numpy.ndarray for the correct labels
            y_hat:a numpy.ndarray for the predicted labels
            pos_label: str or int, the class on which to report
                    the precision_score (default=1)
        Return:
            The recall score as a float.
            None on any error.
        Raises:
            This function should not raise any Exception.
        """
        try:
            if not isinstance(y, np.ndarray) \
                    or not isinstance(y_hat, np.ndarray):
                return None
            if y.shape != y_hat.shape:
                return None
            if y.size == 0 or y_hat.size == 0:
                return None
            if not isinstance(pos_label, (int, str)):
                return None
            tp = np.sum(np.logical_and(y == pos_label, y == y_hat))
            fn = np.sum(np.logical_and(y == pos_label, y_hat != pos_label))
            return tp / (tp + fn)

        except Exception:
            return None

    def f1_score_(self, y, y_hat, pos_label=1):
        """
        Compute the f1 score.
        F1 score combines precision and recall in one single measure.
        You use the F1 score when want to control both
        False positives and False negatives.
        Args:
            y: a numpy.ndarray for the correct labels
            y_hat: a numpy.ndarray for the predicted labels
            pos_label: str or int, the class on which to report
                    the precision_score (default=1)
        Returns:
            The f1 score as a float.
            None on any error.
        Raises:
            This function should not raise any Exception.
        """
        try:
            if not isinstance(y, np.ndarray) \
                    or not isinstance(y_hat, np.ndarray):
                return None
            if y.shape != y_hat.shape:
                return None
            if y.size == 0 or y_hat.size == 0:
                return None
            if not isinstance(pos_label, (int, str)):
                return None
            precision = self.precision_score_(y, y_hat, pos_label)
            recall = self.recall_score_(y, y_hat, pos_label)
            return 2 * (precision * recall) / (precision + recall)

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
            plt.text(0.6, 0.10, f"final loss = {self.losses[-1]:.2f}",
                     transform=plt.gca().transAxes,
                     fontsize=11, verticalalignment='top', color="red")

            # Plot the F1 scores
            ax2 = ax1.twinx()
            ax2.plot(self.f1_scores, color="blue", label="F1 score")
            ax2.legend(["F1 score"], loc="center right",
                       bbox_to_anchor=(0.95, 0.55))
            ax2.set_ylabel("F1 score", color="blue")
            for tl in ax2.get_yticklabels():
                tl.set_color("blue")
            plt.text(0.6, 0.925, f"final F1-score = {self.f1_scores[-1]:.2f}",
                     transform=plt.gca().transAxes,
                     fontsize=11, verticalalignment='top', color="blue")

            plt.title("Metrics evolution computed during training")
            plt.grid(linestyle=':', linewidth=0.5)
            plt.show()
        except Exception:
            return None

    def confusion_matrix_(
            self,
            y_true,
            y_hat,
            labels=None,
            df_option=False,
            display=True):
        """
        Compute confusion matrix to evaluate the accuracy of a classification.
        Args:
            y: a numpy.array for the correct labels
            y_hat: a numpy.array for the predicted labels
            labels: optional, a list of labels to index the matrix.
                    This may be used to reorder or select a subset of labels.
                    (default=None)
            df_option: optional, if set to True the function will return a
                       pandas DataFrame instead of a numpy array.
                       (default=False)
        Return:
            The confusion matrix as a numpy array or a pandas DataFrame
            according to df_option value.
            None if any error.
        Raises:
            This function should not raise any Exception.
        """
        try:
            if not isinstance(y_true, np.ndarray) \
                    or not isinstance(y_hat, np.ndarray):
                print("y_true or y_hat is not a np.ndarray")
                return None
            if y_true.shape != y_hat.shape:
                print("y_true and y_hat must have the same shape")
                return None
            if y_true.size == 0 or y_hat.size == 0:
                print("y_true or y_hat is empty")
                return None
            if labels is None:
                labels = np.unique(np.concatenate((y_true, y_hat)))
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    cm[i, j] = np.where((y_true == labels[i])
                                        & (y_hat == labels[j]))[0].shape[0]
            if df_option:
                cm = DataFrame(cm, index=labels, columns=labels)
            if display:
                print(cm)
            return cm

        except Exception as err:
            print("Error: confusion_matrix_", err)
            return None

    def knn_predict(self, x, i, j, nb_neighbors):
        without_nan_col = np.delete(x, j, 1)
        truth_nan = np.isnan(x[i])
        nan_col = [ind for ind, x in enumerate(truth_nan) if x]
        without_nan_col = np.delete(x, nan_col, 1)
        neighbors = []
        distance = 0.0

        for i_ in range(without_nan_col.shape[0]):
            distance = 0.0
            for j_ in range(without_nan_col.shape[1]):
                distance += np.square(without_nan_col[i_, j_] -
                                      without_nan_col[i, j_])

            if (i_ != i and (not np.isnan(distance))):
                neighbors.append([np.sqrt(distance), i_])

        indmin = np.array(neighbors)[:, 0].argsort()[0:nb_neighbors]

        moy = 0.
        superind = np.array([x[neighbors[y][1], j] for y in indmin])
        superind = superind[~np.isnan(superind)]
        moy = sum(superind) / len(superind)
        return moy

    def knn_imputer(self, x: np.ndarray, nb_neighbors=1):
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if np.isnan(x[i, j]):
                    x[i, j] = self.knn_predict(x, i, j, nb_neighbors)
        return x
