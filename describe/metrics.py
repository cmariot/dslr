import numpy as np


class TinyStatistician:
    """
    Descriptive statistics class for continuous data.
    """

    # Metrics list : (https://en.wikipedia.org/wiki/Descriptive_statistics)
    # Continuous data metrics ideas :
    # Center
    #   Mean                        OK
    #   Median                      OK
    #   Mode                        OK
    # Dispersion
    #   Average absolute deviation  OK
    #   Coefficient of variation    OK
    #   Interquartile range         OK
    #   Percentile                  OK
    #   Range                       OK
    #   Standard deviation          OK
    #   Variance                    OK
    # Shape
    #   Central limit theorem
    #   Moments
    #   Skewness

    def check_arg(func):
        """
        Decorator to check the argument type.
        """
        def wrapper(x):
            try:
                if isinstance(x, np.ndarray):
                    if (len(x.shape) != 1):
                        return None
                    if (x.dtype != np.int64 and x.dtype != np.float64):
                        return None
                    return func(x)
                return None
            except Exception:
                return None
        return wrapper

    @check_arg
    def count(x):
        try:
            ret = 0
            for _ in x:
                ret += 1
            return ret
        except Exception:
            return None

    @check_arg
    def mean(x):
        try:
            if len(x) == 0:
                return None
            ret = 0
            for value in x:
                ret += value
            return float(ret / len(x))
        except Exception:
            return None

    @check_arg
    def var(x):
        try:
            lng = TinyStatistician.count(x)
            if lng == 0:
                return None
            mean = TinyStatistician.mean(x)
            ret = 0
            for value in x:
                ret += (value - mean) ** 2
            return ret / (lng - 1)
        except Exception:
            return None

    @check_arg
    def std(x):
        try:
            var = TinyStatistician.var(x)
            if var is None:
                return None
            return var ** 0.5
        except Exception:
            return None

    @check_arg
    def min(x):
        try:
            mini = x[0]
            for x in x:
                if x < mini:
                    mini = x
            return mini
        except Exception:
            return None

    def percentile(x, p):
        try:
            if not isinstance(x, np.ndarray):
                return None
            if len(x.shape) != 1:
                return None
            if x.dtype != np.int64 and x.dtype != np.float64:
                return None
            if not type(p) is int:
                return None
            lng = len(x)
            if lng == 0:
                return None
            if p > 100 or p < 0:
                return None
            if p == 0:
                return (x[0])
            if p == 100:
                return (x[lng - 1])
            sort = sorted(x)
            obs = (lng - 1) * (p / 100)
            ent = int(obs)
            weight = obs - ent
            return float(sort[ent] + (weight * (sort[ent + 1] - sort[ent])))
        except Exception:
            return None

    @check_arg
    def perc25(x):
        return TinyStatistician.percentile(x, 25)

    @check_arg
    def perc50(x):
        return TinyStatistician.percentile(x, 50)

    @check_arg
    def perc75(x):
        return TinyStatistician.percentile(x, 75)

    @check_arg
    def max(x):
        try:
            maxi = x[0]
            for x in x:
                if x > maxi:
                    maxi = x
            return maxi
        except Exception:
            return None

    @check_arg
    def iqr(x):
        """
        Interquartile range.
        """
        try:
            q1 = TinyStatistician.perc25(x)
            q3 = TinyStatistician.perc75(x)
            if q1 is None or q3 is None:
                return None
            return q3 - q1
        except Exception:
            return None

    @check_arg
    def range(x):
        """
        Range.
        """
        try:
            mini = TinyStatistician.min(x)
            maxi = TinyStatistician.max(x)
            if mini is None or maxi is None:
                return None
            return maxi - mini
        except Exception:
            return None

    @check_arg
    def cv(x):
        """
        Coefficient of variation.
        """
        try:
            std = TinyStatistician.std(x)
            mean = TinyStatistician.mean(x)
            if std is None or mean is None:
                return None
            return std / mean
        except Exception:
            return None

    @check_arg
    def aad(x):
        """
        Average absolute deviation.
        """
        try:
            mean = TinyStatistician.mean(x)
            if mean is None:
                return None
            ret = 0
            for value in x:
                ret += abs(value - mean)
            return ret / len(x)
        except Exception:
            return None

    @check_arg
    def mode(x, split=10):
        """
        If we split x in 'split' tresholds,
        which one is the most frequent.
        """
        x_sorted = sorted(x)
        x_min = TinyStatistician.min(x)
        x_max = TinyStatistician.max(x)
        x_range = x_max - x_min
        x_step = x_range / split
        x_mode = x_min
        x_mode_count = 0
        for _ in range(split):
            x_mode_count_tmp = 0
            for value in x_sorted:
                if value >= x_mode and value < x_mode + x_step:
                    x_mode_count_tmp += 1
            if x_mode_count_tmp > x_mode_count:
                x_mode_count = x_mode_count_tmp
                x_mode = x_mode + x_step
        if x_mode_count == 1:
            # There is no mode
            return None
        return x_mode
