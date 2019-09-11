import math

import pywt


def calculate_mean(window):
    return sum(window) / (len(window) * 1.)


def calculate_energy(window):
    energy = 0.
    for i in window:
        power = i * i if i != 0 else 0.000000000000001
        energy += power
    return energy


def calculate_entropy(window):
    entropy = 0.
    for i in window:
        power = i * i if i != 0 else 0.000000000000001
        v = power * math.log(power)
        entropy += v

    return (-1) * entropy


def calculate_power(energy, size):
    m = 1.0 / size
    return m * energy


def calculate_standard_deviation(mean, window):
    return math.sqrt(sum(list(map(lambda x: (x - mean) * (x - mean), window))) / (len(window) * 1.))


def calculate_amr(window):
    mean = calculate_mean(window)
    return list(map(lambda x: x - mean, window))


def calculate_dwt(data):
    _, detail_level0 = pywt.dwt(data, 'db4')
    return detail_level0[::2]
