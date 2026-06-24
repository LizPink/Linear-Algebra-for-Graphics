import numpy as np


def vector_length(v: np.ndarray) -> float:
    """Return the Euclidean length of a vector."""
    return float(np.sqrt(np.sum(v ** 2)))


def normalize(v: np.ndarray) -> np.ndarray:
    """Return the unit vector in the same direction as v."""
    length = vector_length(v)

    if length == 0:
        raise ValueError("Cannot normalize a zero vector.")

    return v / length


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Return the Euclidean distance between two points."""
    return vector_length(p2 - p1)