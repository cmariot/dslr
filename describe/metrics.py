import numpy as np


class TinyStatistician:

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
