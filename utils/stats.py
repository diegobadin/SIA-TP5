import math
def fmean(values):
    """Return the arithmetic mean of a sequence of numbers as float."""
    total = 0.0
    count = 0
    for v in values:
        total += float(v)
        count += 1
    return total / count if count else 0.0


def pstdev(values):
    """Population standard deviation (denominator = N)."""
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return 0.0
    mu = fmean(vals)
    var = 0.0
    for v in vals:
        diff = v - mu
        var += diff * diff
    var /= n
    return math.sqrt(var)


