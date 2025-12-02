import numpy as np


def sample_power_law(xmin, xmax, n, size):
    # Sample from power law x^n with limits xmin, xmax
    # The parameter size can be an integer or an array for N-dim sampling

    r = np.random.random_sample(size)
    return (-(xmin ** (1 + n) * (-1 + r)) + xmax ** (1 + n) * r) ** (1 / (1 + n))


def sample(fun, MINIMUM_X, MAXIMUM_X, size, strict_size=True):
    mult_fact = 16
    while True:
        xs = np.random.uniform(MINIMUM_X, MAXIMUM_X, size=mult_fact * size)
        cs = np.random.uniform(0, 1, size=mult_fact * size)
        mask = fun(xs) > cs
        masked = xs[mask]
        if len(masked) > size:
            return masked[:size] if strict_size else masked
        else:
            mult_fact *= 2
