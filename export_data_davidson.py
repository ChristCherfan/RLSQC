#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 19:25:20 2026

@author: cesar
"""

import numpy as np
from scipy.sparse import csr_matrix
import os

def export_sparse_matrix_structure(H: csr_matrix, output_dir='.'):
    assert H.shape[0] == H.shape[1], "Matrix must be square"
    N = H.shape[0]

    # Get all row counts at once
    row_nonzero_counts = np.diff(H.indptr)
    Kmax = row_nonzero_counts.max()

    # Preallocate padded arrays
    indices_matrix = np.full((N, Kmax + 1), -1, dtype=int)  # +1 for count
    values_matrix = np.zeros((N, Kmax), dtype=float)

    for i in range(N):
        start = H.indptr[i]
        end = H.indptr[i + 1]
        cols = H.indices[start:end]
        vals = H.data[start:end]

        k = len(cols)
        indices_matrix[i, 0] = k
        indices_matrix[i, 1:k+1] = cols + 1  # 1-based indexing
        values_matrix[i, :k] = vals

    # Save files
    np.savetxt(os.path.join(output_dir, "indices.txt"), indices_matrix, fmt='%d')
    np.savetxt(os.path.join(output_dir, "values.txt"), values_matrix, fmt='%.8f')
    with open(os.path.join(output_dir, "sparsity.txt"), 'w') as f:
        f.write(f"{Kmax}\n")

