import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import combinations

def ev_zeroXings(SC, U, thresh=None):
    """
    Computes the zero crossings for all eigenvectors in SC.
    This function calculates the min-cut/max-flow of the graph.
    
    Parameters:
    SC (numpy.ndarray): The adjacency matrix (nroi x nroi).
    U (numpy.ndarray): The matrix of eigenvectors (nroi x n).
    thresh (float or numpy.ndarray, optional): Threshold for eigenvector values. Default is None.
    
    Returns:
    numpy.ndarray: Zero crossings for each eigenvector.
    """
    nroi = len(SC)

    if thresh is None:
        thresh = np.zeros(nroi)
    elif isinstance(thresh, (float, int)) and thresh >= 1:
        thresh = np.percentile(np.abs(U)**2, thresh, axis=0)
    else:
        thresh = np.ones(nroi) * thresh
    
    zeroX = np.zeros(nroi)
    
    # Generate all combinations of indices (pairs of nodes)
    combos = list(combinations(range(nroi), 2))
    
    # Iterate over each eigenvector
    for k in range(nroi):
        zeroXk = np.zeros(len(combos))

        V = U[:, k]
        V[np.abs(V) ** 2 < thresh[k]] = 0  # Apply thresholding

        # Iterate over all combinations of nodes
        for n, (i, j) in enumerate(combos):
            sgn_vi_x_vj = np.sign(V[i] * V[j])  # Sign of the product

            if sgn_vi_x_vj == -1:
                zeroXk[n] = SC[i, j]

        zeroX[k] = 0.5 * np.sum(zeroXk)

    return zeroX

def zerocrossrate(signal):
    # Count the number of times the signal crosses zero
    return np.sum(np.diff(np.sign(signal)) != 0)
import numpy as np
from scipy.optimize import linear_sum_assignment

def match_eigs(U, V, unmatched_cost=1000):
    """
    Matches two sets of orthonormal eigenvectors, replicating MATLAB's `matchpairs`.

    Args:
    - U (numpy.ndarray): N x N matrix with eigenvectors as columns.
    - V (numpy.ndarray): N x N matrix with eigenvectors as columns.
    - unmatched_cost (float): Cost of leaving an item unmatched.

    Returns:
    - U_sigma (numpy.ndarray): N x N matrix of eigenvectors from U matched to V.
    - sigma (numpy.ndarray): N x 1 vector corresponding to the best permutation.
    """
    # Ensure U and V are square and of the same size
    assert U.shape == V.shape, 'Matrices U and V must be the same size and square.'

    # Calculate the cost matrix as 1 minus the absolute value of the dot products
    cost_matrix = 1 - np.abs(np.dot(U.T, V))

    # Number of rows and columns in the cost matrix
    num_rows, num_cols = cost_matrix.shape

    # Augment the cost matrix to handle unmatched costs
    augmented_matrix = np.full((num_rows + num_cols, num_cols + num_rows), unmatched_cost)
    augmented_matrix[:num_rows, :num_cols] = cost_matrix

    # Solve the assignment problem on the augmented matrix
    row_ind, col_ind = linear_sum_assignment(augmented_matrix)

    # Extract valid matches (ignore dummy assignments)
    valid_assignments = [
        (r, c) for r, c in zip(row_ind, col_ind)
        if r < num_rows and c < num_cols
    ]

    # Create sigma as a mapping of rows to their matched columns
    sigma = np.full(num_rows, -1, dtype=int)  # Initialize all as unmatched (-1)
    for r, c in valid_assignments:
        sigma[r] = c

    # Rearrange U according to sigma to get U_sigma
    U_sigma = U[:, sigma[sigma >= 0]]  # Use only valid matches

    return U_sigma, sigma



def perm_len(v):
    """
    Computes the element-wise permutation length and the total permutation length.
    
    Args:
    - v (numpy.ndarray): The input permutation vector (1-dimensional).
    
    Returns:
    - element_len (numpy.ndarray): A vector where each element is its length in the permutation.
    - len_v (float): The total length of the permutation (sum of element_len / 2).
    """
    N = len(v)
    element_len = np.zeros(N)

    for i in range(N):
        if v[i] == i + 1:  # Since Python uses 0-indexing, compare with i+1
            continue
        else:
            element_len[i] = abs(v[i] - (i + 1))

    len_v = np.sum(element_len) / 2
    return element_len, len_v

def plot_iqr(ax, X, data, metric='median', cmap=None, fill=True, alpha=0.8):
    """
    Recreates the plot_iqr behavior for Python.
    """
    data_median = np.median(data, axis=0)  # Compute the median along rows
    data_iqr = np.percentile(data, [25, 75], axis=0)  # Compute IQR (25th and 75th percentiles)
    
    ax.plot(data_median, label=metric)
    if fill:
        ax.fill_between(X, data_iqr[0], data_iqr[1], alpha=alpha)
        
        


def matchpairs(U_slice, SC_U_consensus, unmatched_cost):
    """
    Perform optimal matching between two matrices and reorder the input slice.

    Args:
    - U_slice (numpy.ndarray): A 2D slice of the matrix U_all (shape: M x N).
    - SC_U_consensus (numpy.ndarray): A 2D matrix to match with (shape: M x N).
    - unmatched_cost (float): Cost of leaving an item unmatched.

    Returns:
    - U_matched (numpy.ndarray): The reordered U_slice matrix based on the matching.
    - matched_order (numpy.ndarray): The matched order indices.
    """
    # Ensure the matrices have the same shape along the first dimension
    assert U_slice.shape[0] == SC_U_consensus.shape[0], "Row dimensions must match!"

    # Compute the cost matrix as the dissimilarity between columns
    num_cols_U = U_slice.shape[1]
    num_cols_SC = SC_U_consensus.shape[1]

    cost_matrix = np.zeros((num_cols_U, num_cols_SC))
    for i in range(num_cols_U):
        for j in range(num_cols_SC):
            # Compute dissimilarity (e.g., Euclidean distance or any other metric)
            cost_matrix[i, j] = np.linalg.norm(U_slice[:, i] - SC_U_consensus[:, j])

    # Determine the size of the augmented matrix
    max_dim = max(num_cols_U, num_cols_SC)
    augmented_matrix = np.full((max_dim, max_dim), unmatched_cost)

    # Fill in the original cost matrix
    augmented_matrix[:num_cols_U, :num_cols_SC] = cost_matrix

    # Solve the assignment problem on the augmented matrix
    row_ind, col_ind = linear_sum_assignment(augmented_matrix)

    # Filter out dummy assignments
    matches = [
        (r, c) for r, c in zip(row_ind, col_ind)
        if r < num_cols_U and c < num_cols_SC
    ]

    # Extract the matching order
    matched_order = np.array([c for r, c in matches])

    # Reorder the U_slice matrix based on the matched order
    U_matched = U_slice[:, matched_order]

    return U_matched, matched_order
