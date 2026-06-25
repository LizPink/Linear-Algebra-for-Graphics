import numpy as np

def squared_length(v):
    """计算向量长度的平方。"""
    v = np.asarray(v, dtype=float)
    return v @ v

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


def local_point_to_world(p_local: np.ndarray, origin_world: np.ndarray, basis: np.ndarray,) -> np.ndarray:
    """Convert a point from local space to world space."""
    return origin_world + basis @ p_local


def local_vector_to_world(v_local: np.ndarray, basis: np.ndarray,) -> np.ndarray:
    """Convert a vector from local space to world space."""
    return basis @ v_local




def cos_angle_between(u, v):
    """计算两个非零向量夹角的 cos(theta)。"""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    denominator = vector_length(u) * vector_length(v)

    if denominator == 0:
        raise ValueError("Cannot compute angle with the zero vector.")

    cos_theta = (u @ v) / denominator

    return np.clip(cos_theta, -1.0, 1.0)


def angle_between(u, v, degrees=True):
    """计算两个非零向量之间的夹角。"""
    theta = np.arccos(cos_angle_between(u, v))

    if degrees:
        return np.degrees(theta)

    return theta


def is_target_in_fov(forward, to_target, fov_degrees):
    """判断目标方向是否在视野角内。"""
    forward_hat = normalize(forward)
    to_target_hat = normalize(to_target)

    cos_theta = forward_hat @ to_target_hat
    threshold = np.cos(np.radians(fov_degrees / 2.0))

    return cos_theta >= threshold


def lambert_intensity(normal, light_direction):
    """计算 Lambert 漫反射强度。"""
    n = normalize(normal)
    l = normalize(light_direction)

    return max(0.0, n @ l)