**Solution:**

Part 1:

```python
import numpy as np

theta = np.deg2rad(45)
R = np.array([[np.cos(theta), -np.sin(theta), 0],
              [np.sin(theta),  np.cos(theta), 0],
              [            0,              0, 1]])

print("R times the x-axis:", R @ [1, 0, 0])
print("R times the y-axis:", R @ [0, 1, 0])
print("R times a 45 degree vector:", R @ [1, 1, 0])
```

Part 2:

Since multiplying a vector by $R$ rotates it 45 degrees, multiplying the result
by $R$ again should result in the original vector being rotated 90 degrees.
Matrix multiplication is associative, which means that $R(Rv) = (RR)v$, so
$S = RR$ should rotate vectors by 90 degrees around the z-axis.

```python
S = R @ R
S @ [1, 0, 0]
```

Part 3:

```python
print("R @ z-axis:", R @ [0, 0, 1])
```

R rotates both the x and y axes, but not the z-axis.

Part 4:

Looking at the documentation of `eig`, we see that it returns two values:
a 1D array of eigenvalues, and a 2D array where each column contains the
eigenvector corresponding to each eigenvalue.

```python
np.linalg.eig(R)
```

In addition to some complex-valued eigenvalues and vectors, we see the value 1
associated with the vector $\left[0, 0, 1\right]^T$.
