import numpy as np
import numba
import time
import scipy

# Helper functions for Least Squares Approximation: Random Projection and Sketching
@numba.jit(nopython=True)
def hamming_weight_array(x):
    count_array = np.zeros_like(x, dtype=np.int8)
    for index in np.ndindex(x.shape):
        count = 0
        value = x[index]
        while value:
            count += value & 1
            value >>= 1
        count_array[index] = count
    return count_array


@numba.jit(nopython=True)
def vectorized_hadamard_rows_with_D(row_indices, n):
    output_rows = np.empty((len(row_indices), n), dtype=np.int8)
    # D = 2 * np.random.randint(0, 2, size=n) - 1
    indices_j = np.arange(n)
    # Compute the Hadamard matrix rows for the given indices
    for count, i in enumerate(row_indices):
        # Compute bitwise AND for every pair i and all j indices
        binary_dot_products = np.bitwise_and(i, indices_j)
        # Compute the Hadamard matrix row entries, then multiply -1 with 0.5 probability
        output_rows[count] = hamming_weight_array(binary_dot_products) * (2 * np.random.randint(0, 2, size=1) - 1)
    return (-1) ** output_rows


# Helper functions for Least Squares Approximation: Random Sampling
def generate_random_projection(r_1, r_2):
    # Generates the randomized unstructured projection (P in our paper)
    return np.random.choice([-np.sqrt(3/r_1), np.sqrt(3/r_1), 0], size=(r_1, r_2), p=[1/6, 1/6, 2/3])


@numba.jit(nopython=True)
def sample_indices(row_indices, n):
    sampled_indices = np.zeros(n, dtype=np.int8)
    for i in range(n):
        random_value = np.random.random()
        cum_sum = 0.0
        for j, weight in enumerate(row_indices):
            cum_sum += weight
            if random_value < cum_sum:
                sampled_indices[i] = j
                break 
    return sampled_indices


def sketch_and_solve(A, b, r, direct=True, precondition_with_QR=False, **kwargs):
    """
    Parameters:
        A: matrix of size m x n
        b: vector of size m (we want to solve Ax = b)
        r: sketch size (dimension we are projecting to)
        direct: whether to use a direct solver or an iterative solver to solve the least squares problem
        precondition_with_QR: whether to precondition the least squares problem with the sketch matrix (for MINRES)
    
    Returns:
        solution to the least squares problem
    """
    m = A.shape[0]
    rand_rows = np.random.choice(m, r, replace=True)
    SH_mD = vectorized_hadamard_rows_with_D(rand_rows, m)
    # Multiply the rows by the Hadamard matrix
    if direct:
        return scipy.linalg.lstsq(SH_mD @ A, SH_mD @ b)[0]
    else:
        if precondition_with_QR:
            _, R = np.linalg.qr(SH_mD @ A)
            minres_sol = scipy.sparse.linalg.minres((SH_mD @ A).T @ (SH_mD @ A) @ R, (SH_mD @ A).T @ SH_mD @ b, **kwargs)[0]
            return np.linalg.solve(R, minres_sol)
        return scipy.sparse.linalg.minres((SH_mD @ A).T @ (SH_mD @ A), (SH_mD @ A).T @ SH_mD @ b, **kwargs)[0]


def sample_and_solve(A, b, r_1, r_2, k, direct=True, precondition_with_QR=False, **kwargs):
    """
    Parameters:
        A: matrix of size m x n
        b: vector of size m (we want to solve Ax = b)
        r_1: sketch size
        r_2: sketch size
        k: number of samples to take for our sketch
        direct: whether to use a direct solver or an iterative solver to solve the least squares problem
        precondition_with_QR: whether to precondition the least squares problem with the sketch matrix (for MINRES)
    
    Returns:
        solution to the least squares problem
    """
    m = A.shape[0]
    rand_rows = np.random.choice(m, r_1, replace=True)
    P = generate_random_projection(r_1, r_2)
    SH_mD = vectorized_hadamard_rows_with_D(rand_rows, m)
    approximate_lev = np.sum((A @ np.linalg.pinv(SH_mD @ A) @ P) ** 2, axis=1)
    approximate_lev /= np.sum(approximate_lev)
    sampled_indices = sample_indices(approximate_lev, k)
    approximate_lev *= k
    sampled_sketch_A = A[sampled_indices] * approximate_lev[sampled_indices, None]
    sampled_sketch_b = b[sampled_indices] * approximate_lev[sampled_indices]
    if direct:
        return scipy.linalg.lstsq(sampled_sketch_A, sampled_sketch_b)[0]
    else:
        if precondition_with_QR:
            _, R = np.linalg.qr(sampled_sketch_A)
            minres_sol = scipy.sparse.linalg.minres(sampled_sketch_A.T @ sampled_sketch_A @ R, sampled_sketch_A.T @ sampled_sketch_b, **kwargs)[0]
            return np.linalg.solve(R, minres_sol)
        return scipy.sparse.linalg.minres(sampled_sketch_A.T @ sampled_sketch_A, sampled_sketch_A.T @ sampled_sketch_b, **kwargs)[0]