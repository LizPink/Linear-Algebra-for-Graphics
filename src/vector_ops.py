from __future__ import annotations

import numpy as np


# =============================================================================
# Basic Utilities
# =============================================================================

def as_float_array(x):
    """把输入转换为 float 类型的 NumPy array。"""
    return np.asarray(x, dtype=float)


def check_same_shape(*arrays):
    """检查多个 array 是否具有相同 shape。"""
    shapes = [arr.shape for arr in arrays]

    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError(f"All inputs must have the same shape, got {shapes}.")


def check_nonzero_vector(v, name="vector", atol=1e-12):
    """检查向量是否不是 zero vector。"""
    if np.isclose(squared_length(v), 0.0, atol=atol):
        raise ValueError(f"{name} must not be the zero vector.")


# =============================================================================
# Dot Product, Length, Distance
# =============================================================================

def dot_product(u, v):
    """计算两个向量的 dot product。

    Parameters
    ----------
    u, v : array-like
        两个 shape 相同的向量。

    Returns
    -------
    float
        u · v
    """
    u = as_float_array(u)
    v = as_float_array(v)

    check_same_shape(u, v)

    return float(u @ v)


def squared_length(v):
    """计算向量长度的平方。

    Parameters
    ----------
    v : array-like
        输入向量。

    Returns
    -------
    float
        向量长度的平方。
    """
    v = as_float_array(v)

    return float(v @ v)


def vector_length(v):
    """计算向量长度。

    Parameters
    ----------
    v : array-like
        输入向量。

    Returns
    -------
    float
        向量长度。
    """
    return float(np.sqrt(squared_length(v)))


def squared_distance(p1, p2):
    """计算两点之间距离的平方。

    Parameters
    ----------
    p1, p2 : array-like
        两个 shape 相同的点。

    Returns
    -------
    float
        两点之间距离的平方。
    """
    p1 = as_float_array(p1)
    p2 = as_float_array(p2)

    check_same_shape(p1, p2)

    d = p2 - p1

    return squared_length(d)


def distance(p1, p2):
    """计算两点之间的欧几里得距离。

    Parameters
    ----------
    p1, p2 : array-like
        两个 shape 相同的点。

    Returns
    -------
    float
        两点之间距离。
    """
    return float(np.sqrt(squared_distance(p1, p2)))


def normalize(v, atol=1e-12):
    """把非零向量 normalized 成单位向量。

    Parameters
    ----------
    v : array-like
        输入向量。
    atol : float
        判断 zero vector 的容忍误差。

    Returns
    -------
    np.ndarray
        单位向量。

    Raises
    ------
    ValueError
        如果 v 是 zero vector。
    """
    v = as_float_array(v)
    length = vector_length(v)

    if np.isclose(length, 0.0, atol=atol):
        raise ValueError("Cannot normalize the zero vector.")

    return v / length


# =============================================================================
# Coordinate Systems and Basis
# =============================================================================

def basis_matrix(*basis_vectors):
    """把 basis vectors 按列组成 basis matrix。

    Parameters
    ----------
    *basis_vectors : array-like
        一组 basis vectors。

    Returns
    -------
    np.ndarray
        basis matrix。
    """
    if len(basis_vectors) == 0:
        raise ValueError("At least one basis vector is required.")

    vectors = [as_float_array(v) for v in basis_vectors]

    first_shape = vectors[0].shape

    if not all(v.shape == first_shape for v in vectors):
        raise ValueError("All basis vectors must have the same shape.")

    return np.column_stack(vectors)


def coordinates_to_standard(B, v_B):
    """把某组 basis 下的坐标转换到 standard coordinates。

    Parameters
    ----------
    B : array-like
        basis matrix，basis vectors 作为列。
    v_B : array-like
        v 在 basis B 下的坐标。

    Returns
    -------
    np.ndarray
        v 在 standard basis 下的坐标。
    """
    B = as_float_array(B)
    v_B = as_float_array(v_B)

    return B @ v_B


def standard_to_coordinates(B, v_E):
    """把 standard coordinates 反解为某组 basis 下的坐标。

    Parameters
    ----------
    B : array-like
        basis matrix，basis vectors 作为列。
    v_E : array-like
        v 在 standard basis 下的坐标。

    Returns
    -------
    np.ndarray
        v 在 basis B 下的坐标。
    """
    B = as_float_array(B)
    v_E = as_float_array(v_E)

    return np.linalg.solve(B, v_E)


# =============================================================================
# Local Space and World Space
# =============================================================================

def local_point_to_world(p_local, origin_world, basis):
    """把 local point 转换到 world space。

    Parameters
    ----------
    p_local : array-like
        点在 local space 中的坐标。
    origin_world : array-like
        local origin 在 world space 中的位置。
    basis : array-like
        local axes 在 world space 中组成的 basis matrix。

    Returns
    -------
    np.ndarray
        点在 world space 中的位置。
    """
    p_local = as_float_array(p_local)
    origin_world = as_float_array(origin_world)
    basis = as_float_array(basis)

    return origin_world + basis @ p_local


def local_vector_to_world(v_local, basis):
    """把 local vector 转换到 world space。

    Parameters
    ----------
    v_local : array-like
        向量在 local space 中的坐标。
    basis : array-like
        local axes 在 world space 中组成的 basis matrix。

    Returns
    -------
    np.ndarray
        向量在 world space 中的坐标。
    """
    v_local = as_float_array(v_local)
    basis = as_float_array(basis)

    return basis @ v_local


# =============================================================================
# Movement
# =============================================================================

def move_point(p, direction, speed, delta_time):
    """沿 direction 移动点 p。

    Parameters
    ----------
    p : array-like
        当前点的位置。
    direction : array-like
        移动方向，不要求已经 normalized。
    speed : float
        移动速度。
    delta_time : float
        时间步长。

    Returns
    -------
    np.ndarray
        移动后的点。
    """
    p = as_float_array(p)
    direction_hat = normalize(direction)

    return p + delta_time * speed * direction_hat


def local_direction_to_world(direction_local, basis):
    """把 local direction 转换成 world direction。

    Parameters
    ----------
    direction_local : array-like
        local space 中的方向。
    basis : array-like
        local axes 在 world space 中组成的 basis matrix。

    Returns
    -------
    np.ndarray
        world space 中的方向。
    """
    return local_vector_to_world(direction_local, basis)


def move_point_with_local_direction(
    p_world,
    direction_local,
    basis,
    speed,
    delta_time,
):
    """使用 local direction 移动物体的 world position。


    Parameters
    ----------
    p_world : array-like
        当前 world position。
    direction_local : array-like
        local space 中的移动方向。
    basis : array-like
        local axes 在 world space 中组成的 basis matrix。
    speed : float
        移动速度。
    delta_time : float
        时间步长。

    Returns
    -------
    np.ndarray
        移动后的 world position。
    """
    direction_world = local_direction_to_world(direction_local, basis)

    return move_point(p_world, direction_world, speed, delta_time)


# =============================================================================
# Angles
# =============================================================================

def cos_angle_between(u, v):
    """计算两个非零向量之间夹角的 cos(theta)。

    Parameters
    ----------
    u, v : array-like
        两个 shape 相同的非零向量。

    Returns
    -------
    float
        cos(theta)，被 clip 到 [-1, 1]。
    """
    u = as_float_array(u)
    v = as_float_array(v)

    check_same_shape(u, v)

    u_len = vector_length(u)
    v_len = vector_length(v)

    if np.isclose(u_len, 0.0) or np.isclose(v_len, 0.0):
        raise ValueError("Cannot compute angle with the zero vector.")

    cos_theta = dot_product(u, v) / (u_len * v_len)

    return float(np.clip(cos_theta, -1.0, 1.0))


def angle_between(u, v, degrees=True):
    """计算两个非零向量之间的夹角。

    Parameters
    ----------
    u, v : array-like
        两个 shape 相同的非零向量。
    degrees : bool
        如果为 True，返回角度制。
        如果为 False，返回弧度制。

    Returns
    -------
    float
        两个向量之间的夹角。
    """
    theta = np.arccos(cos_angle_between(u, v))

    if degrees:
        return float(np.degrees(theta))

    return float(theta)


# =============================================================================
# Orthogonality and Orthonormality
# =============================================================================

def is_orthogonal(u, v, atol=1e-9):
    """判断两个向量是否近似正交。

    Parameters
    ----------
    u, v : array-like
        两个 shape 相同的向量。
    atol : float
        容忍误差。

    Returns
    -------
    bool
        是否近似正交。
    """
    return bool(np.isclose(dot_product(u, v), 0.0, atol=atol))


def is_unit_vector(v, atol=1e-9):
    """判断一个向量是否近似为单位向量。

    Parameters
    ----------
    v : array-like
        输入向量。
    atol : float
        容忍误差。

    Returns
    -------
    bool
        是否近似为单位向量。
    """
    return bool(np.isclose(vector_length(v), 1.0, atol=atol))


def is_orthonormal_frame(r, u, f, atol=1e-9):
    """判断三个方向是否构成近似 orthonormal frame。

    Parameters
    ----------
    r, u, f : array-like
        right, up, forward 三个方向。
    atol : float
        容忍误差。

    Returns
    -------
    bool
        是否构成近似 orthonormal frame。
    """
    return (
        is_orthogonal(r, u, atol=atol)
        and is_orthogonal(u, f, atol=atol)
        and is_orthogonal(f, r, atol=atol)
        and is_unit_vector(r, atol=atol)
        and is_unit_vector(u, atol=atol)
        and is_unit_vector(f, atol=atol)
    )


# =============================================================================
# Projection and Decomposition
# =============================================================================

def scalar_projection(a, b):
    """计算 a 在 b 方向上的 signed scalar projection。

    Parameters
    ----------
    a : array-like
        被投影的向量。
    b : array-like
        投影方向。

    Returns
    -------
    float
        a 在 b 方向上的 signed length。
    """
    a = as_float_array(a)
    b = as_float_array(b)

    check_same_shape(a, b)

    b_hat = normalize(b)

    return dot_product(a, b_hat)


def project_vector(a, b):
    """计算 a 在 b 方向上的 vector projection。

    Parameters
    ----------
    a : array-like
        被投影的向量。
    b : array-like
        投影方向，不能是 zero vector。

    Returns
    -------
    np.ndarray
        a 在 b 方向上的平行分量。
    """
    a = as_float_array(a)
    b = as_float_array(b)

    check_same_shape(a, b)
    check_nonzero_vector(b, name="projection direction")

    return (dot_product(a, b) / dot_product(b, b)) * b


def reject_vector(a, b):
    """计算 a 垂直于 b 的剩余分量。

      Parameters
    ----------
    a : array-like
        被分解的向量。
    b : array-like
        投影方向，不能是 zero vector。

    Returns
    -------
    np.ndarray
        a 垂直于 b 的分量。
    """
    a = as_float_array(a)

    return a - project_vector(a, b)


def decompose_vector(a, b):
    """把 a 分解为沿 b 的平行分量和垂直于 b 的分量。

    Parameters
    ----------
    a : array-like
        被分解的向量。
    b : array-like
        投影方向，不能是 zero vector。

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        parallel component, perpendicular component
    """
    a = as_float_array(a)

    a_parallel = project_vector(a, b)
    a_perp = a - a_parallel

    return a_parallel, a_perp


# =============================================================================
# Point to Line
# =============================================================================

def closest_point_on_line(p, p0, d):
    """计算点 p 到直线 p0 + t d 的最近点。

    Parameters
    ----------
    p : array-like
        待测点。
    p0 : array-like
        直线上的一点。
    d : array-like
        直线方向，不能是 zero vector。

    Returns
    -------
    np.ndarray
        直线上距离 p 最近的点。
    """
    p = as_float_array(p)
    p0 = as_float_array(p0)
    d = as_float_array(d)

    check_same_shape(p, p0, d)
    check_nonzero_vector(d, name="line direction")

    return p0 + project_vector(p - p0, d)


def distance_point_to_line(p, p0, d):
    """计算点 p 到直线 p0 + t d 的距离。

    Parameters
    ----------
    p : array-like
        待测点。
    p0 : array-like
        直线上的一点。
    d : array-like
        直线方向，不能是 zero vector。

    Returns
    -------
    float
        点到直线的距离。
    """
    p = as_float_array(p)
    p_closest = closest_point_on_line(p, p0, d)

    return vector_length(p - p_closest)


# =============================================================================
# Point to Plane
# =============================================================================

def signed_distance_point_to_plane(p, p0, n):
    """计算点 p 到平面的 signed distance。

    Parameters
    ----------
    p : array-like
        待测点。
    p0 : array-like
        平面上的一点。
    n : array-like
        平面法线，不能是 zero vector。

    Returns
    -------
    float
        点到平面的 signed distance。
    """
    p = as_float_array(p)
    p0 = as_float_array(p0)
    n = as_float_array(n)

    check_same_shape(p, p0, n)
    check_nonzero_vector(n, name="plane normal")

    n_hat = normalize(n)

    return dot_product(p - p0, n_hat)


def distance_point_to_plane(p, p0, n):
    """计算点 p 到平面的非负距离。

    Parameters
    ----------
    p : array-like
        待测点。
    p0 : array-like
        平面上的一点。
    n : array-like
        平面法线，不能是 zero vector。

    Returns
    -------
    float
        点到平面的非负距离。
    """
    return abs(signed_distance_point_to_plane(p, p0, n))


def closest_point_on_plane(p, p0, n):
    """计算点 p 到平面的最近点。

    Parameters
    ----------
    p : array-like
        待测点。
    p0 : array-like
        平面上的一点。
    n : array-like
        平面法线，不能是 zero vector。

    Returns
    -------
    np.ndarray
        平面上距离 p 最近的点。
    """
    p = as_float_array(p)
    p0 = as_float_array(p0)
    n = as_float_array(n)

    check_same_shape(p, p0, n)
    check_nonzero_vector(n, name="plane normal")

    n_hat = normalize(n)
    signed_distance = signed_distance_point_to_plane(p, p0, n)

    return p - signed_distance * n_hat


# =============================================================================
# Graphics Applications
# =============================================================================

def is_target_in_fov(forward, to_target, fov_degrees):
    """判断目标方向是否在视野角内。


    Parameters
    ----------
    forward : array-like
        角色或相机的 forward direction。
    to_target : array-like
        从角色或相机指向目标的方向。
    fov_degrees : float
        完整视野角，单位为 degrees。

    Returns
    -------
    bool
        目标是否在视野范围内。
    """
    forward_hat = normalize(forward)
    to_target_hat = normalize(to_target)

    cos_theta = dot_product(forward_hat, to_target_hat)
    threshold = np.cos(np.radians(fov_degrees / 2.0))

    return bool(cos_theta >= threshold)


def lambert_intensity(normal, light_direction):
    """计算 Lambert 漫反射强度。

    数学公式：

    I = max(0, n · l)

    注意：

    normal 和 light_direction 都会先 normalize。

    Parameters
    ----------
    normal : array-like
        表面法线方向。
    light_direction : array-like
        光照方向。

    Returns
    -------
    float
        Lambert 漫反射强度。
    """
    n = normalize(normal)
    l = normalize(light_direction)

    return float(max(0.0, dot_product(n, l)))


def camera_depth(camera_pos, object_pos, camera_forward):
    """计算物体相对相机 forward direction 的 signed depth。

    数学公式：

    depth = (object_pos - camera_pos) · camera_forward_hat

    Parameters
    ----------
    camera_pos : array-like
        相机位置。
    object_pos : array-like
        物体位置。
    camera_forward : array-like
        相机 forward direction。

    Returns
    -------
    float
        signed depth。
    """
    camera_pos = as_float_array(camera_pos)
    object_pos = as_float_array(object_pos)
    camera_forward_hat = normalize(camera_forward)

    check_same_shape(camera_pos, object_pos, camera_forward_hat)

    return dot_product(object_pos - camera_pos, camera_forward_hat)


def project_point_to_ground(p, ground_point, ground_normal):
    """把点投影到地面平面上。

    本质上是 closest_point_on_plane 的语义包装。

    Parameters
    ----------
    p : array-like
        待投影点。
    ground_point : array-like
        地面平面上的一点。
    ground_normal : array-like
        地面法线。

    Returns
    -------
    np.ndarray
        点在地面平面上的投影。
    """
    return closest_point_on_plane(p, ground_point, ground_normal)


# ---------------------------------------------------------------------
# Chapter 03: Cross Product, Orientation, and Surface Normals
# ---------------------------------------------------------------------
def _as_vec3(x, name="vector"):
    """Convert input to a 3D float NumPy vector."""
    x = np.asarray(x, dtype=float)

    if x.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {x.shape}.")

    return x

def cross_product(u, v):
    """Compute the cross product of two 3D vectors."""
    u = _as_vec3(u, "u")
    v = _as_vec3(v, "v")

    return np.cross(u, v)


def cross_magnitude(u, v):
    """Compute the magnitude of the cross product."""
    return np.linalg.norm(cross_product(u, v))


def parallelogram_area(u, v):
    """Compute the area of the parallelogram spanned by u and v."""
    return cross_magnitude(u, v)


def triangle_raw_normal(p0, p1, p2):
    """Compute the raw normal of a triangle."""
    p0 = _as_vec3(p0, "p0")
    p1 = _as_vec3(p1, "p1")
    p2 = _as_vec3(p2, "p2")

    e1 = p1 - p0
    e2 = p2 - p0

    return np.cross(e1, e2)


def is_degenerate_triangle(p0, p1, p2, atol=1e-9):
    """Return True if the triangle is degenerate."""
    n_raw = triangle_raw_normal(p0, p1, p2)
    area_twice = np.linalg.norm(n_raw)

    return np.isclose(area_twice, 0.0, atol=atol)


def triangle_normal(p0, p1, p2, atol=1e-9):
    """Compute the unit normal of a triangle."""
    n_raw = triangle_raw_normal(p0, p1, p2)
    length = np.linalg.norm(n_raw)

    if np.isclose(length, 0.0, atol=atol):
        raise ValueError("Degenerate triangle: normal is undefined.")

    return n_raw / length


def triangle_area(p0, p1, p2):
    """Compute the area of a triangle."""
    n_raw = triangle_raw_normal(p0, p1, p2)

    return 0.5 * np.linalg.norm(n_raw)


def is_triangle_normal_valid(p0, p1, p2, atol=1e-9):
    """Return True if the triangle has a valid normal."""
    return not is_degenerate_triangle(p0, p1, p2, atol=atol)


def is_cross_product_orthogonal(u, v, atol=1e-9):
    """Check whether u x v is orthogonal to both u and v."""
    u = _as_vec3(u, "u")
    v = _as_vec3(v, "v")

    w = np.cross(u, v)

    return (
        np.isclose(w @ u, 0.0, atol=atol)
        and np.isclose(w @ v, 0.0, atol=atol)
    )


def normal_from_edges(e1, e2, normalize_result=True, atol=1e-9):
    """Compute a normal from two triangle edges."""
    e1 = _as_vec3(e1, "e1")
    e2 = _as_vec3(e2, "e2")

    n_raw = np.cross(e1, e2)

    if not normalize_result:
        return n_raw

    length = np.linalg.norm(n_raw)

    if np.isclose(length, 0.0, atol=atol):
        raise ValueError("Edges are parallel or degenerate: normal is undefined.")

    return n_raw / length


def are_opposite_normals(n1, n2, atol=1e-9):
    """Check whether two normals point in opposite directions."""
    n1 = normalize(n1, atol=atol)
    n2 = normalize(n2, atol=atol)

    return np.allclose(n1, -n2, atol=atol)


def does_winding_flip_normal(p0, p1, p2, atol=1e-9):
    """Check whether swapping p1 and p2 flips the triangle normal."""
    n = triangle_normal(p0, p1, p2, atol=atol)
    n_flipped = triangle_normal(p0, p2, p1, atol=atol)

    return are_opposite_normals(n, n_flipped, atol=atol)


def triangle_centroid(p0, p1, p2):
    """Compute the centroid of a triangle."""
    p0 = _as_vec3(p0, "p0")
    p1 = _as_vec3(p1, "p1")
    p2 = _as_vec3(p2, "p2")

    return (p0 + p1 + p2) / 3.0


def view_direction(surface_point, camera_position, atol=1e-9):
    """Compute the unit direction from a surface point to a camera."""
    surface_point = _as_vec3(surface_point, "surface_point")
    camera_position = _as_vec3(camera_position, "camera_position")

    return normalize(camera_position - surface_point, atol=atol)


def facing_score(normal, view_dir, atol=1e-9):
    """Compute the alignment between a normal and a view direction."""
    normal = normalize(normal, atol=atol)
    view_dir = normalize(view_dir, atol=atol)

    return normal @ view_dir


def is_front_facing(p0, p1, p2, camera_position, atol=1e-9):
    """Return True if the triangle is front-facing under the current convention."""
    n = triangle_normal(p0, p1, p2, atol=atol)
    p_surface = triangle_centroid(p0, p1, p2)
    v = view_direction(p_surface, camera_position, atol=atol)

    return facing_score(n, v, atol=atol) > 0.0


def is_back_facing(p0, p1, p2, camera_position, atol=1e-9):
    """Return True if the triangle is back-facing under the current convention."""
    return not is_front_facing(p0, p1, p2, camera_position, atol=atol)