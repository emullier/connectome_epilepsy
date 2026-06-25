%% MATLAB Code: Harmonic Similarity - MATLAB Alignment vs Python Alignment
% Figure 1: Harmonic similarity using MATLAB-native alignment methods (2x2)
% Figure 2: Harmonic similarity using Python alignment methods (2x2)
% Figure 3: Python vs MATLAB comparison overlay (1x3)
%
% Alignment methods:
%   1. Before alignment (baseline)
%   2. Orthogonal Procrustes: SVD-based rotation (MATLAB: svd)
%   3. Generalized Procrustes: rotation + scaling (MATLAB: procrustes)
%   4. Hungarian Matching: column reordering (MATLAB: matchpairs / fallback)
%
% Prerequisite: run export_eigenvectors_for_matlab.py to generate
%   DATA/eigendata_for_matlab.mat  (real HC and IND eigenvectors + Python alignments)
%   DATA/eeg_for_matlab.mat        (EEG inputs for MATLAB SDI computation)

clear all; close all; clc;

% =========================================================================
% CONFIGURATION
% =========================================================================
project_root = pwd;
output_dir = fullfile(project_root, 'FIGURES');
if ~exist(output_dir, 'dir'); mkdir(output_dir); end

colors = [
    31,  119, 180;   % Blue   – before alignment
    255, 127, 14;    % Orange – Orthogonal Procrustes
    44,  160, 44;    % Green  – Generalized Procrustes
    214, 39,  40;    % Red    – Hungarian
] / 255;

% =========================================================================
% DATA LOADING
% =========================================================================
eigendata_path = fullfile(project_root, 'DATA', 'eigendata_for_matlab.mat');
if ~exist(eigendata_path, 'file')
    error('eigendata_for_matlab.mat not found. Run export_eigenvectors_for_matlab.py first.');
end
eigendata = load(eigendata_path);

Q_ref    = eigendata.Q_ref;   % HC reference eigenvectors  (118 x 118)
Q_ind    = eigendata.Q_ind;   % IND eigenvectors, unaligned

% Python-computed alignments (from scipy / gsp.match_eigenvectors)
Q_ind_ortho_py    = eigendata.Q_ind_ortho;    % scipy.linalg.orthogonal_procrustes
Q_ind_rotated_py  = eigendata.Q_ind_rotated;  % scipy.spatial.procrustes
Q_ind_matched_py  = eigendata.Q_ind_matched;  % gsp.match_eigenvectors (Hungarian)

n_rois      = size(Q_ref, 1);   % 118
n_harmonics = size(Q_ref, 2);   % 118
harmonic_indices = 1:n_harmonics;

fprintf('Loaded eigenvector matrices: Q_ref (%dx%d), Q_ind (%dx%d)\n', ...
        n_rois, n_harmonics, size(Q_ind,1), size(Q_ind,2));

% =========================================================================
% MATLAB-NATIVE ALIGNMENTS
% =========================================================================

% 1. Orthogonal Procrustes: R = argmin ||Q_ref - Q_ind*R||_F, R'R = I
fprintf('Computing Orthogonal Procrustes (MATLAB)...\n');
[U_op, ~, V_op] = svd(Q_ind' * Q_ref);
R_ortho = U_op * V_op';
Q_ind_ortho_matlab = Q_ind * R_ortho;

% 2. Generalized Procrustes: rotation + isotropic scaling + translation
fprintf('Computing Generalized Procrustes (MATLAB)...\n');
[~, Q_ind_rotated_matlab] = procrustes(Q_ref, Q_ind);

% 3. Hungarian matching: column permutation maximising |correlation|
fprintf('Computing Hungarian matching (MATLAB)...\n');
cost_matrix = zeros(n_harmonics, n_harmonics);
for ii = 1:n_harmonics
    for jj = 1:n_harmonics
        cost_matrix(ii, jj) = -abs(corr(Q_ref(:, ii), Q_ind(:, jj)));
    end
end
[~, col_idx] = hungarian(cost_matrix);
Q_ind_matched_matlab = Q_ind(:, col_idx);

% =========================================================================
% COMPUTE HARMONIC SIMILARITY (|correlation| per eigenmode)
% =========================================================================
sim_raw             = compute_harmonic_similarity(Q_ref, Q_ind);
% MATLAB alignments
sim_ortho_matlab    = compute_harmonic_similarity(Q_ref, Q_ind_ortho_matlab);
sim_rotated_matlab  = compute_harmonic_similarity(Q_ref, Q_ind_rotated_matlab);
sim_matched_matlab  = compute_harmonic_similarity(Q_ref, Q_ind_matched_matlab);
% Python alignments
sim_ortho_py        = compute_harmonic_similarity(Q_ref, Q_ind_ortho_py);
sim_rotated_py      = compute_harmonic_similarity(Q_ref, Q_ind_rotated_py);
sim_matched_py      = compute_harmonic_similarity(Q_ref, Q_ind_matched_py);

% Stats
[mean_raw,    std_raw]    = compute_stats(sim_raw);
[mean_ortho,  std_ortho]  = compute_stats(sim_ortho_matlab);
[mean_rot,    std_rot]    = compute_stats(sim_rotated_matlab);
[mean_match,  std_match]  = compute_stats(sim_matched_matlab);

% Wilcoxon tests vs raw
[p_ortho,  W_ortho]  = wilcoxon_test(sim_ortho_matlab,   sim_raw);
[p_rot,    W_rot]    = wilcoxon_test(sim_rotated_matlab,  sim_raw);
[p_match,  W_match]  = wilcoxon_test(sim_matched_matlab,  sim_raw);

% Stats for Python alignments
[mean_ortho_py,  std_ortho_py]  = compute_stats(sim_ortho_py);
[mean_rot_py,    std_rot_py]    = compute_stats(sim_rotated_py);
[mean_match_py,  std_match_py]  = compute_stats(sim_matched_py);

[p_ortho_py,  W_ortho_py]  = wilcoxon_test(sim_ortho_py,   sim_raw);
[p_rot_py,    W_rot_py]    = wilcoxon_test(sim_rotated_py,  sim_raw);
[p_match_py,  W_match_py]  = wilcoxon_test(sim_matched_py,  sim_raw);

% =========================================================================
% FIGURE 1: MATLAB Alignment – Harmonic Similarity (2x2)
% =========================================================================
fprintf('\nGenerating Figure 1: MATLAB Alignment Harmonic Similarity\n');

fig1 = figure('Position', [100, 100, 1200, 800]);
panels_mat = {sim_raw, sim_ortho_matlab, sim_rotated_matlab, sim_matched_matlab};
panel_means_mat = [mean_raw, mean_ortho,    mean_rot,    mean_match];
panel_stds_mat  = [std_raw,  std_ortho,     std_rot,     std_match];
panel_titles_mat = { ...
    'A. Before alignment', ...
    sprintf('B. Orthogonal Procrustes\nvs before, p=%.3e %s', p_ortho, get_significance_label(p_ortho)), ...
    sprintf('C. Generalized Procrustes\nvs before, p=%.3e %s', p_rot,   get_significance_label(p_rot)), ...
    sprintf('D. Hungarian matching\nvs before, p=%.3e %s',     p_match, get_significance_label(p_match))};

for i = 1:4
    subplot(2, 2, i);
    hold on;
    fill_between_plot(harmonic_indices, ...
                      panels_mat{i} - panel_stds_mat(i), ...
                      panels_mat{i} + panel_stds_mat(i), colors(i,:), 0.15);
    plot(harmonic_indices, panels_mat{i}, 'LineWidth', 2.5, 'Color', colors(i,:));
    yline(panel_means_mat(i), ':', 'LineWidth', 1.5, 'Color', colors(i,:), 'Alpha', 0.8);
    xlabel('Eigenmode', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('|Correlation|', 'FontSize', 12, 'FontWeight', 'bold');
    ylim([0, 1.05]); xlim([1, n_harmonics]); xticks(0:20:n_harmonics);
    title(sprintf('%s\n(mean=%.3f±%.3f)', panel_titles_mat{i}, panel_means_mat(i), panel_stds_mat(i)), ...
          'FontSize', 11, 'FontWeight', 'bold');
    set(gca, 'FontSize', 11); box off; grid on;
    ax = gca; ax.XAxis.LineWidth = 1.5; ax.YAxis.LineWidth = 1.5;
end
sgtitle('MATLAB Alignment: Harmonic Similarity (SC IND vs SC HC)', ...
        'FontSize', 14, 'FontWeight', 'bold');
fig1_path = fullfile(output_dir, 'Fig1_MATLAB_alignment_harmonic_similarity.png');
saveas(fig1, fig1_path);
fprintf('Figure 1 saved as ''%s''\n', fig1_path);

% Print MATLAB stats
fprintf('\n--- MATLAB alignments ---\n');
fprintf('%-28s  %8s  %10s  %8s  %5s\n', 'Method', 'mean', 'W', 'p', 'sig');
fprintf('%s\n', repmat('-', 1, 65));
fprintf('%-28s  %8.4f\n', 'Before alignment', mean_raw);
fprintf('%-28s  %8.4f  %10.1f  %8.2e  %5s\n', 'Ortho. Procrustes (MATLAB)', mean_ortho,  W_ortho,  p_ortho,  get_significance_label(p_ortho));
fprintf('%-28s  %8.4f  %10.1f  %8.2e  %5s\n', 'Gen. Procrustes (MATLAB)',   mean_rot,    W_rot,    p_rot,    get_significance_label(p_rot));
fprintf('%-28s  %8.4f  %10.1f  %8.2e  %5s\n', 'Hungarian (MATLAB)',         mean_match,  W_match,  p_match,  get_significance_label(p_match));

% =========================================================================
% FIGURE 2: Python Alignment – Harmonic Similarity (2x2)
% =========================================================================
fprintf('\nGenerating Figure 2: Python Alignment Harmonic Similarity\n');

fig2 = figure('Position', [100, 100, 1200, 800]);
panels_py = {sim_raw, sim_ortho_py, sim_rotated_py, sim_matched_py};
panel_means_py = [mean_raw, mean_ortho_py,  mean_rot_py,  mean_match_py];
panel_stds_py  = [std_raw,  std_ortho_py,   std_rot_py,   std_match_py];
panel_titles_py = { ...
    'A. Before alignment', ...
    sprintf('B. Orthogonal Procrustes\nvs before, p=%.3e %s', p_ortho_py, get_significance_label(p_ortho_py)), ...
    sprintf('C. Generalized Procrustes\nvs before, p=%.3e %s', p_rot_py,   get_significance_label(p_rot_py)), ...
    sprintf('D. Hungarian matching\nvs before, p=%.3e %s',     p_match_py, get_significance_label(p_match_py))};

for i = 1:4
    subplot(2, 2, i);
    hold on;
    fill_between_plot(harmonic_indices, ...
                      panels_py{i} - panel_stds_py(i), ...
                      panels_py{i} + panel_stds_py(i), colors(i,:), 0.15);
    plot(harmonic_indices, panels_py{i}, 'LineWidth', 2.5, 'Color', colors(i,:));
    yline(panel_means_py(i), ':', 'LineWidth', 1.5, 'Color', colors(i,:), 'Alpha', 0.8);
    xlabel('Eigenmode', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('|Correlation|', 'FontSize', 12, 'FontWeight', 'bold');
    ylim([0, 1.05]); xlim([1, n_harmonics]); xticks(0:20:n_harmonics);
    title(sprintf('%s\n(mean=%.3f±%.3f)', panel_titles_py{i}, panel_means_py(i), panel_stds_py(i)), ...
          'FontSize', 11, 'FontWeight', 'bold');
    set(gca, 'FontSize', 11); box off; grid on;
    ax = gca; ax.XAxis.LineWidth = 1.5; ax.YAxis.LineWidth = 1.5;
end
sgtitle('Python Alignment: Harmonic Similarity (SC IND vs SC HC)', ...
        'FontSize', 14, 'FontWeight', 'bold');
fig2_path = fullfile(output_dir, 'Fig2_Python_alignment_harmonic_similarity.png');
saveas(fig2, fig2_path);
fprintf('Figure 2 saved as ''%s''\n', fig2_path);

% Print Python stats
fprintf('\n--- Python alignments ---\n');
fprintf('%-28s  %8s  %10s  %8s  %5s\n', 'Method', 'mean', 'W', 'p', 'sig');
fprintf('%s\n', repmat('-', 1, 65));
fprintf('%-28s  %8.4f\n', 'Before alignment', mean_raw);
fprintf('%-28s  %8.4f  %10.1f  %8.2e  %5s\n', 'Ortho. Procrustes (Python)', mean_ortho_py,  W_ortho_py,  p_ortho_py,  get_significance_label(p_ortho_py));
fprintf('%-28s  %8.4f  %10.1f  %8.2e  %5s\n', 'Gen. Procrustes (Python)',   mean_rot_py,    W_rot_py,    p_rot_py,    get_significance_label(p_rot_py));
fprintf('%-28s  %8.4f  %10.1f  %8.2e  %5s\n', 'Hungarian (Python)',         mean_match_py,  W_match_py,  p_match_py,  get_significance_label(p_match_py));

% =========================================================================
% FIGURE 3: Python vs MATLAB – Harmonic Similarity Overlay (1x3)
% =========================================================================
fprintf('\nGenerating Figure 3: Python vs MATLAB Alignment Comparison\n');

py_sims      = {sim_ortho_py,     sim_rotated_py,     sim_matched_py};
mat_sims     = {sim_ortho_matlab,  sim_rotated_matlab,  sim_matched_matlab};
method_labels = {'Orthogonal Procrustes', 'Generalized Procrustes', 'Hungarian matching'};

fig3 = figure('Position', [100, 100, 1400, 420]);
for i = 1:3
    subplot(1, 3, i);
    hold on;
    plot(harmonic_indices, py_sims{i},  'Color', [0.20 0.50 0.90], 'LineWidth', 2.5, 'DisplayName', 'Python (scipy)');
    plot(harmonic_indices, mat_sims{i}, 'Color', [0.90 0.35 0.15], 'LineWidth', 2.5, 'LineStyle', '--', 'DisplayName', 'MATLAB');
    xlabel('Eigenmode', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('|Correlation|', 'FontSize', 12, 'FontWeight', 'bold');
    ylim([0, 1.05]); xlim([1, n_harmonics]); xticks(0:20:n_harmonics);
    mean_diff = mean(py_sims{i} - mat_sims{i});
    mad_diff  = mean(abs(py_sims{i} - mat_sims{i}));
    title(sprintf('%s\nMean diff = %.4f  |diff| = %.4f', method_labels{i}, mean_diff, mad_diff), ...
          'FontSize', 11, 'FontWeight', 'bold');
    legend('Location', 'northeast', 'FontSize', 9);
    grid on; box off;
    set(gca, 'FontSize', 11);
    ax = gca; ax.XAxis.LineWidth = 1.5; ax.YAxis.LineWidth = 1.5;
end
sgtitle('Python (scipy) vs MATLAB: Harmonic Similarity to HC Reference', ...
        'FontSize', 13, 'FontWeight', 'bold');
fig3_path = fullfile(output_dir, 'Fig3_Python_vs_MATLAB_alignment.png');
saveas(fig3, fig3_path);
fprintf('Figure 3 saved as ''%s''\n', fig3_path);

% =========================================================================
% SDI COMPUTATION IN MATLAB (same EEG data source as Python script)
% =========================================================================
eeg_path = fullfile(project_root, 'DATA', 'eeg_for_matlab.mat');
if ~exist(eeg_path, 'file')
    warning(['eeg_for_matlab.mat not found. Run export_eigenvectors_for_matlab.py first. ' ...
             'Skipping SDI figures.']);
else
    fprintf('\nLoading EEG inputs for SDI from: %s\n', eeg_path);
    eeg_data = load(eeg_path);

    if (~isfield(eeg_data, 'X_RS_cell') && ~isfield(eeg_data, 'X_RS_all')) || ~isfield(eeg_data, 'lat_labels')
        warning('eeg_for_matlab.mat missing required fields (X_RS_cell or X_RS_all, and lat_labels). Skipping SDI figures.');
    else
        if isfield(eeg_data, 'X_RS_cell')
            X_RS_cell = eeg_data.X_RS_cell;  % preferred: variable-size per subject
        else
            % backward compatibility with old fixed-size export format
            X_RS_all_legacy = eeg_data.X_RS_all;
            n_sub_legacy = size(X_RS_all_legacy, 4);
            X_RS_cell = cell(n_sub_legacy, 1);
            for s_legacy = 1:n_sub_legacy
                X_RS_cell{s_legacy, 1} = X_RS_all_legacy(:, :, :, s_legacy);
            end
        end

        lat_labels = eeg_data.lat_labels;

        n_subjects = numel(X_RS_cell);
        idxs_LT = get_lat_indices(lat_labels, 'Ltle');
        idxs_RT = get_lat_indices(lat_labels, 'Rtle');
        fprintf('Found %d LT subjects and %d RT subjects for SDI computation.\n', numel(idxs_LT), numel(idxs_RT));

        method_names_sdi = {'SC_HC_ref', 'SC_IND', 'Ortho_Procrustes', 'Gen_Procrustes', 'Hungarian'};
        method_labels_sdi = {'Before alignment', 'Orthogonal Procrustes', 'Generalized Procrustes', 'Hungarian matching'};
        Q_methods = {Q_ref, Q_ind, Q_ind_ortho_matlab, Q_ind_rotated_matlab, Q_ind_matched_matlab};

        mean_log2_LT = cell(1, numel(Q_methods));
        mean_log2_RT = cell(1, numel(Q_methods));

        fprintf('Computing SDI for each method in MATLAB...\n');
        for m = 1:numel(Q_methods)
            SDI_all = zeros(n_rois, n_subjects);
            for s = 1:n_subjects
                X_RS_sub = X_RS_cell{s};
                SDI_all(:, s) = compute_sdi_subject(X_RS_sub, Q_methods{m});
            end

            SDI_log2 = log2(max(SDI_all, eps));
            if ~isempty(idxs_LT)
                mean_log2_LT{m} = mean(SDI_log2(:, idxs_LT), 2, 'omitnan');
            else
                mean_log2_LT{m} = [];
            end
            if ~isempty(idxs_RT)
                mean_log2_RT{m} = mean(SDI_log2(:, idxs_RT), 2, 'omitnan');
            else
                mean_log2_RT{m} = [];
            end
            fprintf('  SDI computed for %s\n', method_names_sdi{m});
        end

        % LT figure (Fig3b style)
        if ~isempty(mean_log2_LT{1})
            fprintf('Generating SDI LT scatter plots (MATLAB-computed)...\n');
            fig3b = figure('Position', [100, 100, 1000, 1000]);
            comp_idx = [2, 3, 4, 5];  % compare vs SC_HC_ref
            all_vals_lt = [mean_log2_LT{1}(:); mean_log2_LT{2}(:); mean_log2_LT{3}(:); mean_log2_LT{4}(:); mean_log2_LT{5}(:)];
            min_lt = min(all_vals_lt); max_lt = max(all_vals_lt); pad_lt = 0.03 * max(1e-6, max_lt - min_lt);

            for i = 1:4
                subplot(2, 2, i);
                x = mean_log2_LT{1}(:);
                y = mean_log2_LT{comp_idx(i)}(:);
                valid = isfinite(x) & isfinite(y);
                x = x(valid); y = y(valid);

                scatter(x, y, 36, 'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.75);
                hold on;
                plot([min_lt-pad_lt, max_lt+pad_lt], [min_lt-pad_lt, max_lt+pad_lt], '--', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.25);

                if numel(x) >= 3
                    [Rmat, Pmat] = corrcoef(x, y, 'Rows', 'complete');
                    R = Rmat(1, 2); P = Pmat(1, 2);
                else
                    R = NaN; P = NaN;
                end

                title(sprintf('%s\\nr=%.3f, p=%.2e', method_labels_sdi{i}, R, P), 'FontSize', 11, 'FontWeight', 'bold');
                xlabel('log2 SDI SC HC ref (LT)', 'FontSize', 11, 'FontWeight', 'bold');
                ylabel('log2 SDI method (LT)', 'FontSize', 11, 'FontWeight', 'bold');
                xlim([min_lt-pad_lt, max_lt+pad_lt]);
                ylim([min_lt-pad_lt, max_lt+pad_lt]);
                axis square; grid on; box on;
                set(gca, 'FontSize', 10);
            end

            sgtitle('Fig3b-style SDI Scatter (LT, MATLAB-computed)', 'FontSize', 14, 'FontWeight', 'bold');
            fig3b_path = fullfile(output_dir, 'Fig3b_SDI_scatter_LT_matlab_computed.png');
            saveas(fig3b, fig3b_path);
            fprintf('SDI LT figure saved as ''%s''\n', fig3b_path);
        end

        % RT figure (supplementary style)
        if ~isempty(mean_log2_RT{1})
            fprintf('Generating SDI RT scatter plots (MATLAB-computed)...\n');
            figS4 = figure('Position', [120, 120, 1000, 1000]);
            comp_idx = [2, 3, 4, 5];
            all_vals_rt = [mean_log2_RT{1}(:); mean_log2_RT{2}(:); mean_log2_RT{3}(:); mean_log2_RT{4}(:); mean_log2_RT{5}(:)];
            min_rt = min(all_vals_rt); max_rt = max(all_vals_rt); pad_rt = 0.03 * max(1e-6, max_rt - min_rt);

            for i = 1:4
                subplot(2, 2, i);
                x = mean_log2_RT{1}(:);
                y = mean_log2_RT{comp_idx(i)}(:);
                valid = isfinite(x) & isfinite(y);
                x = x(valid); y = y(valid);

                scatter(x, y, 36, 'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.75);
                hold on;
                plot([min_rt-pad_rt, max_rt+pad_rt], [min_rt-pad_rt, max_rt+pad_rt], '--', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.25);

                if numel(x) >= 3
                    [Rmat, Pmat] = corrcoef(x, y, 'Rows', 'complete');
                    R = Rmat(1, 2); P = Pmat(1, 2);
                else
                    R = NaN; P = NaN;
                end

                title(sprintf('%s\\nr=%.3f, p=%.2e', method_labels_sdi{i}, R, P), 'FontSize', 11, 'FontWeight', 'bold');
                xlabel('log2 SDI SC HC ref (RT)', 'FontSize', 11, 'FontWeight', 'bold');
                ylabel('log2 SDI method (RT)', 'FontSize', 11, 'FontWeight', 'bold');
                xlim([min_rt-pad_rt, max_rt+pad_rt]);
                ylim([min_rt-pad_rt, max_rt+pad_rt]);
                axis square; grid on; box on;
                set(gca, 'FontSize', 10);
            end

            sgtitle('SDI Scatter (RT, MATLAB-computed)', 'FontSize', 14, 'FontWeight', 'bold');
            figS4_path = fullfile(output_dir, 'FigS4_SDI_scatter_RT_matlab_computed.png');
            saveas(figS4, figS4_path);
            fprintf('SDI RT figure saved as ''%s''\n', figS4_path);
        end
    end
end

fprintf('\nDone.\n');

%% =========================================================================
%% HELPER FUNCTIONS
%% =========================================================================

function similarity = compute_harmonic_similarity(Q_ref, Q_aligned)
    % Compute correlation-based similarity for each harmonic (column)
    % similarity(i) = 1 - corr_distance(Q_ref(:,i), Q_aligned(:,i))
    
    n_harmonics = size(Q_ref, 2);
    similarity = zeros(n_harmonics, 1);
    
    for i = 1:n_harmonics
        % Correlation distance
        corr_val = abs(corr(Q_ref(:, i), Q_aligned(:, i)));
        % Convert correlation to similarity (1 - distance)
        similarity(i) = corr_val;
    end
    
    % Clamp to [0, 1]
    similarity = max(0, min(1, similarity));
end

function [row_idx, col_idx] = hungarian(cost_matrix)
    % Hungarian Algorithm for bipartite matching using MATLAB built-ins.
    % Solves the linear assignment problem to minimize total cost.
    % cost_matrix(i,j) should be negative correlation (to minimize = maximize corr).
    
    n = size(cost_matrix, 1);
    
    % Try matchpairs (R2020a+, no toolbox needed).
    % matchpairs requires non-negative costs, so shift cost matrix to [0, 2].
    try
        cost_shifted = cost_matrix - min(cost_matrix(:));  % shift to >= 0
        M = matchpairs(cost_shifted, max(cost_shifted(:)) * n + 1);  % penalty > any total cost
        M = sortrows(M, 1);   % ensure sorted by row index
        row_idx = M(:, 1)';
        col_idx = M(:, 2)';
        return;
    catch
    end
    
    % Fallback: true Hungarian algorithm implemented via Jonker-Volgenant style
    % using MATLAB's built-in linear programming (no toolbox needed)
    % We implement the shortest augmenting path Hungarian algorithm.
    col_idx = zeros(1, n);
    u = zeros(1, n+1);   % potential for rows
    v = zeros(1, n+1);   % potential for cols
    p = zeros(1, n+1);   % col -> row assignment (1-indexed, 0 = unassigned)
    way = zeros(1, n+1); % augmenting path
    
    for i = 1:n
        p(1) = i;
        j0 = 1;
        minVal = inf(1, n+1);
        used = false(1, n+1);
        
        while true
            used(j0) = true;
            i0 = p(j0);
            delta = inf;
            j1 = -1;
            for j = 2:n+1
                if ~used(j)
                    cur = cost_matrix(i0, j-1) - u(i0) - v(j);
                    if cur < minVal(j)
                        minVal(j) = cur;
                        way(j) = j0;
                    end
                    if minVal(j) < delta
                        delta = minVal(j);
                        j1 = j;
                    end
                end
            end
            % Update potentials
            for j = 1:n+1
                if used(j)
                    u(p(j)) = u(p(j)) + delta;
                    v(j) = v(j) - delta;
                else
                    minVal(j) = minVal(j) - delta;
                end
            end
            j0 = j1;
            if p(j0) == 0
                break;
            end
        end
        % Augment along path
        while j0 ~= 1
            p(j0) = p(way(j0));
            j0 = way(j0);
        end
    end
    
    % Extract assignment: p(j) = row assigned to col j-1
    col_idx = zeros(1, n);
    for j = 2:n+1
        if p(j) ~= 0
            col_idx(p(j)) = j - 1;
        end
    end
    row_idx = 1:n;
end

function [mean_val, std_val] = compute_stats(data)
    % Compute mean and standard deviation
    mean_val = mean(data);
    std_val = std(data);
end

function [p_value, W_stat] = wilcoxon_test(x, y)
    % Wilcoxon signed-rank test (one-sided: alternative H1: x > y)
    % Returns p-value and test statistic W
    
    % Difference
    diff = x - y;
    
    % Remove zeros
    diff_nz = diff(diff ~= 0);
    
    % If all zeros, return NaN
    if isempty(diff_nz)
        p_value = NaN;
        W_stat = NaN;
        return;
    end
    
    % Ranks of absolute differences
    [~, ranks] = sort(abs(diff_nz));
    ranks(ranks) = 1:length(ranks);  % Assign ranks
    
    % Sum of ranks for positive differences
    W_stat = sum(ranks(diff_nz > 0));
    
    % Approximate p-value using normal distribution
    n = length(diff_nz);
    mu = n * (n + 1) / 4;
    sigma = sqrt(n * (n + 1) * (2*n + 1) / 24);
    z = (W_stat - mu) / sigma;
    
    % One-sided p-value (testing whether x > y)
    p_value = 1 - normcdf(z);  % right tail
end

function sig_label = get_significance_label(p_value)
    % Convert p-value to significance label
    if p_value < 0.001
        sig_label = '***';
    elseif p_value < 0.01
        sig_label = '**';
    elseif p_value < 0.05
        sig_label = '*';
    else
        sig_label = 'n.s.';
    end
end

function fill_between_plot(x, y_lower, y_upper, color, alpha)
    % Create filled area between two curves
    x_fill = [x(:); flipud(x(:))];
    y_fill = [y_lower(:); flipud(y_upper(:))];
    fill(x_fill, y_fill, color, 'EdgeColor', 'none', 'FaceAlpha', alpha);
end

function SDI = compute_sdi_subject(X_RS, scU)
    % MATLAB equivalent of Python gsp.compute_SDI (empirical SDI only)
    xmu = mean(X_RS(:), 'omitnan');
    xsig = std(X_RS(:), 'omitnan');
    if xsig <= eps
        zX_RS = zeros(size(X_RS));
    else
        zX_RS = (X_RS - xmu) ./ xsig;
    end

    [~, ~, Vlow, Vhigh] = get_cutoff_freq_matlab(scU, zX_RS);

    n_rois = size(X_RS, 1);
    n_ep = size(X_RS, 3);
    N_c = zeros(n_rois, n_ep);
    N_d = zeros(n_rois, n_ep);

    for ep = 1:n_ep
        X_hat = scU' * zX_RS(:, :, ep);
        X_c = Vlow * X_hat;
        X_d = Vhigh * X_hat;
        for r = 1:n_rois
            N_c(r, ep) = norm(X_c(r, :));
            N_d(r, ep) = norm(X_d(r, :));
        end
    end

    SDI = mean(N_d, 2, 'omitnan') ./ max(mean(N_c, 2, 'omitnan'), eps);
end

function [PSD, NN, Vlow, Vhigh] = get_cutoff_freq_matlab(sc, data)
    % MATLAB equivalent of Python gsp.get_cutoff_freq
    n_rois = size(sc, 1);
    n_time = size(data, 2);
    n_ep = size(data, 3);

    X_hat_L = zeros(n_rois, n_time, n_ep);
    for ep = 1:n_ep
        X_hat_L(:, :, ep) = sc' * data(:, :, ep);
    end

    pow = abs(X_hat_L).^2;
    PSD = squeeze(mean(pow, 2));
    mPSD = mean(PSD, 2);
    AUCTOT = trapz(mPSD(1:n_rois));

    i = 1;
    AUC = 0;
    while (AUC < AUCTOT/2) && (i < n_rois)
        i = i + 1;
        AUC = trapz(mPSD(1:i));
    end
    NN = i - 1;

    Vlow = zeros(size(sc));
    Vhigh = zeros(size(sc));
    Vhigh(:, NN+1:end) = sc(:, NN+1:end);
    Vlow(:, 1:NN) = sc(:, 1:NN);
end

function idxs = get_lat_indices(lat_labels, target_label)
    % Robust extraction of subject indices from MATLAB-loaded label arrays
    if iscell(lat_labels)
        labels = string(lat_labels);
    elseif isstring(lat_labels)
        labels = lat_labels;
    elseif ischar(lat_labels)
        labels = string(cellstr(lat_labels));
    else
        labels = string(lat_labels(:));
    end
    labels = strip(labels);
    idxs = find(labels == string(target_label));
end

