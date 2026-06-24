# Mathematical Conventions

This document records the mathematical conventions used in this project.

## Vector Convention

This project uses column vectors.

A transformation matrix acts on a vector or point from the left:

$$
p'
=
Mp
$$

## Transformation Order

For column vectors, the rightmost matrix is applied first.

$$
p'
=
TRSp
$$

This means:

1. S is applied first;
2. R is applied second;
3. T is applied last.

## Coordinate System

This project uses a right-handed coordinate system.

The cross product follows the right-hand rule.

## Points and Vectors

A point represents a position in space.

A vector represents a direction, displacement, velocity, normal, or light direction.

In homogeneous coordinates, a point is represented as:

$$
p
=
\begin{bmatrix}
x \\
y \\
z \\
1
\end{bmatrix}
$$

A vector is represented as:

$$
v
=
\begin{bmatrix}
x \\
y \\
z \\
0
\end{bmatrix}
$$

Translation affects points but does not affect vectors.

## NumPy Convention

Use:

```python
M @ v
```

for matrix-vector multiplication.

Use:

```python
M * v
```

only for element-wise multiplication.