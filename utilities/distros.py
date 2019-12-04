#! /usr/bin/env python3

import numpy as np
from collections import OrderedDict

def multivariate_standard_normal(shape):
    return np.random.standard_normal(np.prod(shape)).reshape(shape).astype('f4')

def get_2dcov(theta, sigma_x = 1.0, sigma_y = 1.0):
    a = np.square(np.cos(theta)) / 2 * np.square(sigma_x) + np.square(np.sin(theta)) / 2 * np.square(sigma_y)
    b = np.sin(2 * theta) / 4 * np.square(sigma_x) - np.sin(2 * theta) / 4 * np.square(sigma_y)
    c = np.square(np.sin(theta)) / 2 * np.square(sigma_x) + np.square(np.cos(theta)) / 2 * np.square(sigma_y)

    return np.array([[a, b], [b, c]]).astype('f4')

def get_2dmean(theta, radius = 1.0):
    return np.array([radius * np.cos(theta), radius * np.sin(theta)]).astype('f4')

def mix2d(distributions, sample_size, coefficients = None):
    if coefficients is None:
        coefficients = np.ones(len(distributions), dtype = 'f4')

    coefficients = np.asarray(coefficients)

    coefficients /= coefficients.sum()

    num_distr = len(distributions)
    data = np.zeros((num_distr, sample_size, 2))
    for idx, distr in enumerate(distributions):
        data[idx] = distr["type"](size=(sample_size,), **distr["kwargs"])
    random_idx = np.random.choice(np.arange(num_distr), size=(sample_size,), p=coefficients)

    sample = np.ascontiguousarray(data[random_idx, np.arange(sample_size), :], dtype = 'f4')
    return sample, random_idx

def umbrella_distros(thetas, means, sx, sy, shuffle = False, shifts = None):

    distributions = []

    if shuffle:
        idx = np.random.shuffle(np.arange(thetas.size))
    else:
        idx = np.arange(thetas.size)

    if shifts == None:
        shifts = np.zeros_like(thetas)

    for i in idx:
        distributions.append({"type": np.random.multivariate_normal, "kwargs": {"mean": get_2dmean(thetas[i], radius = means[i] + shifts[i]), "cov": get_2dcov(thetas[i], sigma_x = sx[i], sigma_y = sy[i])}})
        
    return distributions