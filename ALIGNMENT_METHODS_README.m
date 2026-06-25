% =========================================================================
% ALIGNMENT METHODS IMPLEMENTATION SUMMARY
% =========================================================================
%
% This document describes the alignment methods implemented in the MATLAB code
% for aligning eigenvector spaces between two connectome datasets.
%
% =========================================================================
% 1. ORTHOGONAL PROCRUSTES ALIGNMENT
% =========================================================================
%
% Method: Finds the best orthogonal rotation matrix R that minimizes:
%         ||Q_ref - Q_ind * R||_F^2
%
% MATLAB Implementation:
%   [U, S, V] = svd(Q_ref' * Q_ind);
%   R_ortho = V * U';
%   Q_ind_ortho = Q_ind * R_ortho;
%
% Why SVD?
%   - U and V are orthonormal matrices from SVD decomposition
%   - R = V * U' is guaranteed to be orthogonal (det(R) = ±1)
%   - This preserves the norm and orthogonality of eigenvectors
%
% Properties:
%   - Rotation only (no scaling)
%   - Full-space rotation mixing all modes
%   - Optimal in Frobenius norm sense
%
% =========================================================================
% 2. GENERALIZED PROCRUSTES ALIGNMENT (with Scaling)
% =========================================================================
%
% Method: Finds optimal rotation AND scaling that minimizes:
%         ||Q_ref - scale * Q_ind * R||_F^2
%
% MATLAB Implementation:
%   [d, Z, transform] = procrustes(Q_ref, Q_ind);
%   Q_ind_rotated = Z;
%
% What procrustes() does:
%   - Computes SVD of centered, scaled data
%   - Applies rotation AND scaling transformation
%   - d = Procrustes statistic (residual sum of squares)
%   - Z = aligned matrix (Q_ind transformed)
%   - transform.scale = scaling factor
%   - transform.T = rotation matrix
%
% Properties:
%   - Allows both rotation and isotropic scaling
%   - More flexible than orthogonal Procrustes
%   - May distort the eigenspace structure if scale ≠ 1
%
% =========================================================================
% 3. HUNGARIAN MATCHING (Bipartite Assignment)
% =========================================================================
%
% Method: Reorders columns of Q_ind to maximize correlation with Q_ref columns
%         Solves: maximize Σ corr(Q_ref(:,i), Q_ind(:,σ(i)))
%         where σ is a permutation of columns
%
% MATLAB Implementation:
%   cost_matrix(i,j) = -abs(corr(Q_ref(:,i), Q_ind(:,j)));
%   [row_idx, col_idx] = hungarian(cost_matrix);  % Custom function
%   Q_ind_matched = Q_ind(:, col_idx);
%
% How it works:
%   1. Compute pairwise correlation between all eigenvector pairs
%   2. Build cost matrix (negative correlations for minimization)
%   3. Solve assignment problem to find optimal permutation
%   4. Reorder Q_ind columns according to assignment
%
% Notes:
%   - Uses greedy approximation if Optimization Toolbox unavailable
%   - True Hungarian algorithm: O(n³) complexity
%   - For best results, install munkres() from MATLAB File Exchange
%
% Properties:
%   - No rotation or scaling, only column reordering
%   - Preserves eigenvalue magnitudes and orthonormality
%   - More conservative approach compared to rotation methods
%
% =========================================================================
% 4. BASELINE: BEFORE ALIGNMENT
% =========================================================================
%
% Method: No transformation applied
%   Q_ind_raw = Q_ind;
%
% Purpose:
%   - Serves as control/baseline for comparison
%   - Shows effect of misalignment between cohorts
%   - Tests statistical significance of alignment methods
%
% =========================================================================
% SIMILARITY METRICS
% =========================================================================
%
% For each alignment method, the script computes per-harmonic similarity:
%
%   similarity(i) = 1 - corr_distance(Q_ref(:,i), Q_aligned(:,i))
%                 = abs(corr(Q_ref(:,i), Q_aligned(:,i)))
%
% where:
%   - Q_ref(:,i) = i-th eigenvector from reference connectome
%   - Q_aligned(:,i) = i-th eigenvector from aligned method
%   - corr() = Pearson correlation coefficient
%   - abs() = absolute value (to handle sign flips)
%
% =========================================================================
% STATISTICAL TESTING (Wilcoxon Signed-Rank)
% =========================================================================
%
% H0: Aligned methods do NOT improve similarity compared to baseline
% H1: Aligned methods DO improve similarity (one-sided test)
%
% Test Statistic: W = sum of positive difference ranks
%
% Procedure:
%   1. Compute difference: diff = similarity_aligned - similarity_baseline
%   2. Rank absolute differences: |diff(1)|, ..., |diff(n_harmonics)|
%   3. Sum ranks where diff > 0: W = Σ rank(diff > 0)
%   4. Compute p-value using normal approximation (for n > 30)
%   5. One-sided p-value: P(W ≥ W_obs) = 1 - Φ(z)
%
% Result interpretation:
%   - p < 0.05: Significant improvement
%   - p ≥ 0.05: No significant improvement (n.s.)
%   - *** p < 0.001, ** p < 0.01, * p < 0.05
%
% =========================================================================
% SDI (SIGNIFICANT DETECTABILITY INDEX)
% =========================================================================
%
% Definition: Graph signal detectability index measuring signal correlation
% after projection onto eigenspace
%
% Computation:
%   X_proj = Q * (Q' * X)  % Project signal onto eigenspace
%   SDI(roi) = corr(X(roi,:), X_proj(roi,:))  % Per-ROI correlation
%
% Purpose:
%   - Assess how well eigenspace captures signal structure
%   - Compare signal detectability between alignment methods
%   - Validate alignment quality through practical signal processing
%
% =========================================================================
% KEY DIFFERENCES BETWEEN METHODS
% =========================================================================
%
% Method                 | Rotation | Scaling | Reordering | Constraints
% --------                ------    -------   ----------   -----------
% Before Alignment       |    No    |   No    |     No     | Baseline
% Orthogonal Procrustes  |   Yes    |   No    |     No     | det(R)=±1
% Generalized Procrustes |   Yes    |   Yes   |     No     | None
% Hungarian Matching     |    No    |   No    |    Yes     | Bijection
%
% =========================================================================
% USAGE NOTES
% =========================================================================
%
% 1. To use with your actual data:
%    - Load Q_ref and Q_ind from your connectome files
%    - Replace synthetic data generation with actual data loading
%    - Ensure Q matrices are orthonormal (columns are eigenvectors)
%
% 2. For Hungarian algorithm improvement:
%    - Download munkres.m from MATLAB File Exchange
%    - Uncomment the true Hungarian implementation in hungarian.m
%    - Or use assignmentoptimal() if you have Optimization Toolbox
%
% 3. SDI computation:
%    - Replace synthetic X_signal with actual EEG data
%    - X_signal should be (n_rois × time_points) matrix
%    - Can use multiple subjects and average SDI values
%
% 4. Performance considerations:
%    - SVD for Procrustes: O(n³) where n = n_harmonics
%    - Hungarian algorithm: O(n³) with greedy, O(n²·5) with optimal
%    - SDI computation: O(n_rois × n_time × n_harmonics)
%
% =========================================================================
