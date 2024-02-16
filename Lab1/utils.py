"""
Utility functions for Lab1
Author: Kausshik Manojkumar
"""
import numpy as np

def is_in(element, sequence):
    """Similar to (element in sequence), but compares with 'is', not '=='."""
    return any(x is element for x in sequence)

def euclidean_distance(x, y):
    return np.sqrt(sum((_x - _y) ** 2 for _x, _y in zip(x, y)))

def manhattan_distance(x, y):
    return sum(abs(_x - _y) for _x, _y in zip(x, y))

def gaschnig_distance(x, y):
    return max(abs(_x - _y) for _x, _y in zip(x, y))