import numpy as np
from scipy.linalg import hadamard

def hadamard_sketch(r, m):
    # Constructs a Hadamard Sketch Matrix
    # Takes:
    # - r (dimension of )
    # - m (order of Hadamard Matrix)
    # Gives:
    # - Hadamard Sketch

    H = hadamard(2 ** m)
    D = np.diag(np.random.choice([-1, 1], int(2**m)))
    indices = np.random.choice(2 ** m, r)
    S = []
    for index in indices:
        new = np.zeros(2 ** m)
        new[index] = 1
        S.append(new)
    S = np.vstack(S)
    return S @ H @ D

def project_solve(A, b):
    r, m = 10, int(np.ceil(np.log2(A.shape[0])))
    sketch = hadamard_sketch(r, m)
    A_tilde = np.zeros((2 ** m, 2 ** m))
    b_tilde = np.zeros((2 ** m, 1))
    A_tilde[:A.shape[0], :A.shape[1]] = A
    b_tilde[:b.shape[0]] = b
    return (np.linalg.pinv(sketch @ A_tilde) @ (sketch @ b_tilde))[:A.shape[1]] 