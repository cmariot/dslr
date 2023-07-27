import sys
import numpy as np
import math
class TinyStatistician():
    @staticmethod

    def check_type(li):

        niteger = [int, float]
        if type(li) is list:
            for x in li:
                if (not type(x) in niteger):
                    return False
            return True
        if isinstance(li, np.ndarray):
            if (len(li.shape) != 1):
                return False
            if (li.dtype != np.int64 and li.dtype != np.float64):
                return False
            return True
        return False

    def mean(self, li):
        print(li, type(li))
        if not TinyStatistician.check_type(li):
            return None
        if (len(li) == 0):
            return None

        ret = 0
        for x in li:
            ret = ret + x
        return (float(ret / len(li)))

    def median(self, li):
        if not TinyStatistician.check_type(li):
            return None
        lng = len(li)
        if (lng < 1):
            return None
        sort = sorted(li)
        if lng % 2 == 1:
            return (float(sort[int((lng + 1)/ 2 - 1)]))
        else:
            return(float((sort[int(lng / 2 - 1)] + sort[int(lng / 2)]) / 2))

    def quartile(self, li):
        if not TinyStatistician.check_type(li):
            return None
        lng = len(li)
        if (lng < 1): #or 4
            return None
        sort = sorted(li)
        
        r1 = math.ceil(lng / 4)
        r2 = math.ceil(lng / 4 * 3)
        return ([float(sort[int(r1 - 1)]), float(sort[int(r2 - 1)])])

    def percentile(self, x, p):
        if not TinyStatistician.check_type(x):
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

        return (float(sort[ent] + (weight * (sort[ent + 1] - sort[ent]))))


    def var(self, li):
        if not TinyStatistician.check_type(li):
            return None
        lng = len(li)
        if (lng == 0):
            return None
        mean = self.mean(li)

        ret = 0
        for x in li:
            ret = ret + (x - mean)**2
        return (float(round(ret / (lng - 1), 1)))

    def std(self, li):
        if not TinyStatistician.check_type(li):
            return None
        lng = len(li)
        if (lng == 0):
            return None
        mean = self.mean(li)

        ret = 0
        for x in li:
            ret = ret + (x - mean)**2
        return (float(math.sqrt(ret / (lng - 1))))