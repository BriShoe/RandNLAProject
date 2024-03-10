import numpy as np
from sklearn.preprocessing import normalize
import pandas as pd

def sample_len_2(A, s):
    '''
    Samples according to the length-squared sample.
    '''
    probs = np.linalg.norm(A, axis=1) ** 2 / np.linalg.norm(A) ** 2
    R = np.zeros((s, A.shape[1]))
    rows = np.random.choice(A.shape[0], size=s, p=probs)
    for i in range(s):
        R[i, :] = 1/probs[rows[i]] * A[rows[i], :]
    return R

def LRA(eps, A, k, s=None):
    '''
    Gives a Low-Rank Approximation via Randomness.
    '''
    if s is None:
        s = int(np.ceil(2 * np.sqrt(k) * np.linalg.norm(A) ** 4 / eps))
    if s > A.shape[0]:
        print("This will not save time, but we will proceed anyways.")
    R = sample_len_2(A, s)
    U,S,_ = np.linalg.svd(R @ R.T)
    left_singular_vectors = normalize(U @ R, norm="l2")
    projection = np.zeros((A.shape[1], A.shape[1]))
    for i in range(k):
        projection += np.outer(left_singular_vectors[i], left_singular_vectors[i])
    return A @ projection

def LRA_r(A, r):
    '''
    Best Low-Rank Approximation (Frobenius Norm) using SVD.
    '''
    U, S, V = np.linalg.svd(A)
    S = np.diag(S)
    return U[:, :r] @ S[:r, :r] @ V[:r, :]