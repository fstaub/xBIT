import math
import random


# Log and linear distributions; what else?
def LINEAR(min, max, steps=0):
    if steps > 0:
        return [1.0 * min
                + 1.0 * (max - min) * (a - 1) / (1.0 * (steps - 1))
                for a in range(1, steps + 1)]
    else:
        return random.uniform(min, max)


def LINEAR_DIFF(a, b):
    return 1. * abs(a - b)


def LOG(min, max, steps=0):
    if steps > 0:
        return [math.exp(math.log(min)
                + (math.log(max)-math.log(min))*(a-1)/((steps-1)))
                for a in range(1, steps + 1)]
    else:
        return math.exp(
            math.log(min) 
            + (math.log(max) - math.log(min)) * random.uniform(0, 1))


# gauss likelihood: do we need something else?
def gauss(val, m, s):
    return math.exp(-(val-m)**2/(2.*s)**2)


def logL(val, m, s):
    return 1./(1.+(val-m)**2/(2.*s)**2)


def limit_lh(steps):
    if (steps < 1.0e4):
        return 0.1
    elif (steps < 1.0e5):
        return 0.01
    elif (steps < 1.0e6):
        return 0.001
    else:
        return 1.0e-5


def step(x):
    if x > 0.5:
        return 1.
    else:
        return 0.


def exp_safe(x):
    try:
        out = math.exp(x)
    except OverflowError:
        out = float('inf')
    return out


# def dd(a, b):
#     # 'distance' between two points
#     return sum([(abs((x-y)/(x+y))) for x, y in zip(a, b)])
