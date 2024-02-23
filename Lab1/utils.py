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

def index_to_position(index):
        """Convert a 1D index to a 2D position in a 3x3 puzzle."""
        return (index // 3, index % 3)

def manhattan_distance(x, y):
    return sum(abs(_x - _y) for _x, _y in zip(x, y))

def gaschnig_distance(x, y):
    return max(abs(_x - _y) for _x, _y in zip(x, y))

def read_puzzle_state(filename):
    with open(filename, 'r') as f:
        state = []
        for line in f:
            row = line.strip().split()
            state.extend(row)
    #Replace the blank tile with 0
    state = [0 if x == '_' else int(x) for x in state]
    return tuple(state)
