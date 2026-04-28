%% =========================================================
%  FINAL ANALYSIS SCRIPT
%  Pupil-BOLD-NE-ACh Correlation & GLM Analysis
%  =========================================================

%% SECTION 1: Load fMRI data
fmriData = load('/Users/sowinglum/Downloads/neurophysiology/fmri/fMRI.mat');
fn = fieldnames(fmriData);
V = fmriData.(fn{1});           % (102 x 197 x 75 x 205)
[nx, ny, nz, nt] = size(V);
fprintf('fMRI loaded: %d x %d x %d x %d\n', nx, ny, nz, nt)

%% SECTION 2: Load pupil data
pupilData = load('/Users/sowinglum/Downloads/neurophysiology/fmri/pupil 1.mat');
fn = fieldnames(pupilData);
pupil = pupilData.(fn{1});
pupil = pupil(:);               % ensure column vector (205 x 1)
fprintf('Pupil loaded: %d timepoints\n', length(pupil))

%% SECTION 3: Global mean BOLD + z-score + clean
V_2d      = reshape(V, [], nt); %(voxels * 205) 
mean_bold = mean(V_2d, 1)'; 
mean_bold_z = (mean_bold - mean(mean_bold)) / std(mean_bold);
outliers    = abs(mean_bold_z) > 3;
mean_bold_clean = mean_bold_z;
if sum(outliers) > 0
    mean_bold_clean(outliers) = interp1(...
        find(~outliers), mean_bold_z(~outliers), ...
        find(outliers), 'linear');
end
fprintf('BOLD outliers removed: %d\n', sum(outliers))

%% SECTION 4: Pupil z-score + clean
pupil_z = (pupil - mean(pupil)) / std(pupil);
pupil_outliers = abs(pupil_z) > 3;
pupil_clean = pupil_z;
if sum(pupil_outliers) > 0
    pupil_clean(pupil_outliers) = interp1(...
        find(~pupil_outliers), pupil_z(~pupil_outliers), ...
        find(pupil_outliers), 'linear');
end
fprintf('Pupil outliers removed: %d\n', sum(pupil_outliers))

%% SECTION 5: Brain mask
mean_vol  = mean(V, 4);
brain_mask = mean_vol > mean(mean_vol(:)) * 0.3;
mask_vec   = brain_mask(:);
fprintf('Brain voxels: %d / %d total\n', sum(mask_vec), numel(mask_vec))

%% SECTION 6: BOLD-Pupil lag analysis
lags   = -10:10;
r_lags = zeros(size(lags));
for i = 1:length(lags)
    lag = lags(i);
    if lag > 0
        C = corrcoef(mean_bold_clean(lag+1:end), pupil_clean(1:end-lag));
    elseif lag < 0
        C = corrcoef(mean_bold_clean(1:end+lag), pupil_clean(-lag+1:end));
    else
        C = corrcoef(mean_bold_clean, pupil_clean);
    end
    r_lags(i) = C(1,2);
end
[~, max_i] = max(abs(r_lags));
fprintf('Best lag: %d TRs, r = %.4f\n', lags(max_i), r_lags(max_i))

%% SECTION 7: NE and ACh lag analysis
NE_data  = readtable('/Users/sowinglum/Downloads/neurophysiology/NE/pupil_NE.csv');
ACh_data = readtable('/Users/sowinglum/Downloads/neurophysiology/Ach/pupil_ACh.csv');

% Remove NaNs
valid_NE  = ~isnan(NE_data.pupil);
valid_ACh = ~isnan(ACh_data.pupil);

pupil_NE_z  = (NE_data.pupil(valid_NE)  - mean(NE_data.pupil(valid_NE)))  / std(NE_data.pupil(valid_NE));
NE_z        = (NE_data.NE(valid_NE)     - mean(NE_data.NE(valid_NE)))     / std(NE_data.NE(valid_NE));
pupil_ACh_z = (ACh_data.pupil(valid_ACh)- mean(ACh_data.pupil(valid_ACh)))/ std(ACh_data.pupil(valid_ACh));
ACh_z       = (ACh_data.Ach(valid_ACh)  - mean(ACh_data.Ach(valid_ACh)))  / std(ACh_data.Ach(valid_ACh));

lags_samples = -300:300;
lags_sec     = lags_samples * 0.0333;
r_NE         = zeros(size(lags_samples));
r_ACh        = zeros(size(lags_samples));

for i = 1:length(lags_samples)
    lag = lags_samples(i);
    if lag > 0
        C_NE  = corrcoef(NE_z(lag+1:end),  pupil_NE_z(1:end-lag));
        C_ACh = corrcoef(ACh_z(lag+1:end), pupil_ACh_z(1:end-lag));
    elseif lag < 0
        C_NE  = corrcoef(NE_z(1:end+lag),  pupil_NE_z(-lag+1:end));
        C_ACh = corrcoef(ACh_z(1:end+lag), pupil_ACh_z(-lag+1:end));
    else
        C_NE  = corrcoef(NE_z,  pupil_NE_z);
        C_ACh = corrcoef(ACh_z, pupil_ACh_z);
    end
    r_NE(i)  = C_NE(1,2);
    r_ACh(i) = C_ACh(1,2);
end
fprintf('NE-pupil  peak r = %.4f at lag = %.2f s\n', max(abs(r_NE)),  lags_sec(abs(r_NE)==max(abs(r_NE))))
fprintf('ACh-pupil peak r = %.4f at lag = %.2f s\n', max(abs(r_ACh)), lags_sec(abs(r_ACh)==max(abs(r_ACh))))

%% SECTION 8: GLM - Pupil regressor convolved with HRF
addpath('/Users/sowinglum/Documents/MATLAB/spm');
TR      = 1;
hrf     = spm_hrf(TR);
pupil_reg = conv(pupil_clean, hrf);
pupil_reg = pupil_reg(1:nt);
pupil_reg = (pupil_reg - mean(pupil_reg)) / std(pupil_reg);

X = [pupil_reg, ones(nt, 1)];
fprintf('Design matrix: %d x %d\n', size(X,1), size(X,2))

% Z-score each voxel timeseries before GLM
data_2d   = double(reshape(V, [], nt)');    % (205 x voxels)
vox_mean  = mean(data_2d, 1);
vox_std   = std(data_2d, 0, 1);
vox_std(vox_std < 1e-10) = 1;
data_2d_z = (data_2d - vox_mean) ./ vox_std; %(205* voxles) 

fprintf('Running GLM...\n')
betas_z     = X \ data_2d_z;
residuals_z = data_2d_z - X * betas_z;
df          = nt - size(X, 2);
sigma2_z    = sum(residuals_z.^2) / df;
contrast    = [1, 0];
var_beta_z  = (contrast * inv(X'*X) * contrast') .* sigma2_z;
t_map_z     = (contrast * betas_z) ./ sqrt(var_beta_z);

beta_map_z = reshape(betas_z(1,:), [nx, ny, nz]);
t_map_3d_z = reshape(t_map_z,      [nx, ny, nz]);

t_masked_z    = t_map_3d_z;  t_masked_z(~brain_mask)    = NaN;
beta_masked_z = beta_map_z;  beta_masked_z(~brain_mask) = NaN;

fprintf('GLM done. t-map range: %.3f to %.3f\n', ...
    min(t_map_3d_z(:)), max(t_map_3d_z(:)))
fprintf('Voxels |t|>2: %d\n', sum(abs(t_masked_z(:))>2, 'omitnan'))
fprintf('Voxels |t|>3: %d\n', sum(abs(t_masked_z(:))>3, 'omitnan'))

% Find best slice (exclude top/bottom 10)
n_sig_glm = zeros(nz,1);
for z = 10:nz-10
    n_sig_glm(z) = sum(abs(t_masked_z(:,:,z))>2, 'all', 'omitnan');
end
[~, best_z_glm] = max(n_sig_glm);
fprintf('Best GLM slice: z=%d\n', best_z_glm)

%% SECTION 9: Save NIfTI files for SPM
info = niftiinfo('/Users/sowinglum/Documents/MATLAB/spm/matlabbatch/full_fmri_timeseries.nii');
info.Datatype        = 'single';
info.ImageSize       = [nx ny nz];
info.PixelDimensions = info.PixelDimensions(1:3);

% Mean brain
niftiwrite(single(mean_vol), ...
    '/Users/sowinglum/Documents/MATLAB/spm/matlabbatch/mean_brain.nii', info);

%% Save z-scored GLM t-map for SPM overlay
t_nii_z = t_map_3d_z;
t_nii_z(~brain_mask)           = NaN;
t_nii_z(abs(t_map_3d_z) < 2)   = NaN;

niftiwrite(single(t_nii_z), ...
    '/Users/sowinglum/Documents/MATLAB/spm/matlabbatch/pupil_glm_tmap.nii', info);
disp('Saved pupil_glm_tmap.nii')

%% =========================================================
%  FIGURES
%  =========================================================

%% FIGURE 1: Global BOLD vs Pupil timeseries
figure('Position', [100 100 900 350]);
plot(mean_bold_clean, 'b', 'LineWidth', 1.2); hold on;
plot(pupil_clean,     'r', 'LineWidth', 1.2);
legend('Global BOLD (z)', 'Pupil (z)', 'Location', 'northeast');
xlabel('Timepoint (TR = 1s)'); ylabel('Z-score');
title('Global BOLD vs Pupil (r = 0.171, lag = 4 TR)');
xlim([1 205]); box off;

%% FIGURE 2: BOLD-Pupil lag correlation bar chart
figure('Position', [100 100 600 400]);
bar(lags, r_lags, 'FaceColor', [0.3 0.5 0.8], 'EdgeColor', 'none');
hold on;
xline(0, '--k', 'LineWidth', 1.5);
xline(4, '--r', 'LineWidth', 1.5, 'Label', 'optimal lag');
xlabel('Lag (TRs, 1TR = 1s)'); ylabel('Pearson r');
title('BOLD-Pupil Correlation as a Function of Hemodynamic Lag');
ylim([-0.25 0.25]); box off;

%% FIGURE 3: Combined multimodal lag correlation
figure('Position', [100 100 1000 450]);
lags_bold_sec = lags * 1;
plot(lags_bold_sec, r_lags, 'b-o', 'LineWidth', 2, 'MarkerSize', 6); hold on;
plot(lags_sec, r_NE,  'g-',                        'LineWidth', 1.5);
plot(lags_sec, r_ACh, 'color', [0.9 0.6 0],        'LineWidth', 1.5);
xline(0, '--k', 'LineWidth', 1);

[~, ne_i]  = max(abs(r_NE));
[~, ach_i] = max(abs(r_ACh));
scatter(lags_sec(ne_i),  r_NE(ne_i),   80, 'g',         'filled');
scatter(lags_sec(ach_i), r_ACh(ach_i), 80, [0.9 0.6 0], 'filled');
scatter(4, r_lags(lags_bold_sec==4),   80, 'b',         'filled');

text(4.2,  0.19, 'r=0.17, lag=+4s',  'Color', 'b',         'FontSize', 9);
text(8.0,  0.42, 'r=0.40, lag=+10s', 'Color', [0 0.6 0],   'FontSize', 9);
text(-9.5, 0.65, 'r=0.62, lag=-5s',  'Color', [0.8 0.5 0], 'FontSize', 9);

xlim([-10 10]); ylim([-0.7 0.7]);
xlabel('Lag (s)'); ylabel('Pearson r');
title('Pupil-Brain Signal Correlations Across Modalities');
legend('BOLD fMRI', 'NE fiber photometry', 'ACh fiber photometry', ...
    'Location', 'southwest');
box off; grid on;

%% Clean up GLM figure
figure('Position', [100 100 1200 400], 'Color', 'w');

% subplot(1,3,1)
% imagesc(rot90(beta_masked_z(:,:,best_z_glm)));
% colormap(gca, 'hot'); colorbar; axis image;
% clim([-0.3 0.3]); set(gca, 'Color', 'k');
% title(sprintf('Beta map (z=%d)', best_z_glm));
% xlabel('x'); ylabel('y');

subplot(1,3,1)
imagesc(rot90(mean_vol(:,:,best_z_glm)));
colormap(gca, 'gray'); colorbar; axis image;
title(sprintf('Mean brain (z=%d)', best_z_glm));
xlabel('x'); ylabel('y');

subplot(1,3,2)
imagesc(rot90(t_masked_z(:,:,best_z_glm)));
colormap(gca, 'jet'); colorbar; axis image;
clim([-5 5]); set(gca, 'Color', 'k');
title('T-statistic map');
xlabel('x'); ylabel('y');

subplot(1,3,3)
thresh = t_masked_z(:,:,best_z_glm);
thresh(abs(thresh) < 2) = NaN;      % |t|>2 not 3
imagesc(rot90(thresh), 'AlphaData', rot90(~isnan(thresh)));
colormap(gca, 'jet'); colorbar; axis image;
clim([-5 5]); set(gca, 'Color', 'k');
title('Thresholded |t|>2 (p<0.05 uncorrected)');
xlabel('x'); ylabel('y');

sgtitle(sprintf('GLM: Pupil → BOLD (z-scored, HRF convolved, df=%d)', df));

% Count surviving voxels
fprintf('Voxels |t|>2: %d\n', sum(abs(t_masked_z(:))>2, 'omitnan'))
fprintf('Voxels |t|>3: %d\n', sum(abs(t_masked_z(:))>3, 'omitnan'))
%% FIGURE 5: GLM Design Matrix
figure;
imagesc(X); colormap gray; colorbar;
title('GLM Design Matrix');
xlabel('Regressors'); ylabel('Timepoints');
xticks([1 2]); xticklabels({'Pupil (HRF)', 'Intercept'});

%% p-value for BOLD-pupil correlation
n      = 201;
t_stat = 0.1712 * sqrt(n-2) / sqrt(1 - 0.1712^2);
% Replace tcdf with SPM's equivalent: spm_Tcdf(t_statistic, degrees_of_freedom)
p_val = 2 * (1 - spm_Tcdf(abs(t_stat), n-2));
fprintf('BOLD-pupil: r=0.171, t(%d)=%.3f, p=%.4f\n', n-2, t_stat, p_val)



%% Smooth the fMRI data (3x3x3 box = 8 neighbors)
% fprintf('Smoothing fMRI data...\n')
% 
% V_smooth = zeros(size(V), 'single');
% box_kernel = ones(3,3,3) / 27;  % 3x3x3 averaging kernel
% 
% for t = 1:nt
%     vol = single(V(:,:,:,t));
%     V_smooth(:,:,:,t) = imfilter(vol, box_kernel, 'replicate');
% end
% fprintf('Smoothing done.\n')
% 
% %% Rerun GLM on smoothed data
% data_2d_s   = double(reshape(V_smooth, [], nt)');
% vox_mean_s  = mean(data_2d_s, 1);
% vox_std_s   = std(data_2d_s, 0, 1);
% vox_std_s(vox_std_s < 1e-10) = 1;
% data_2d_sz  = (data_2d_s - vox_mean_s) ./ vox_std_s;
% 
% fprintf('Running smoothed GLM...\n')
% betas_s     = X \ data_2d_sz;
% residuals_s = data_2d_sz - X * betas_s;
% sigma2_s    = sum(residuals_s.^2) / df;
% var_beta_s  = (contrast * inv(X'*X) * contrast') .* sigma2_s;
% t_map_s     = (contrast * betas_s) ./ sqrt(var_beta_s);
% 
% beta_map_s  = reshape(betas_s(1,:),  [nx, ny, nz]);
% t_map_3d_s  = reshape(t_map_s,       [nx, ny, nz]);
% 
% t_masked_s  = t_map_3d_s;
% t_masked_s(~brain_mask) = NaN;
% 
% fprintf('Smoothed t-map range: %.3f to %.3f\n', ...
%     min(t_map_3d_s(:)), max(t_map_3d_s(:)))
% fprintf('Voxels |t|>2: %d\n', sum(abs(t_masked_s(:))>2, 'omitnan'))
% fprintf('Voxels |t|>3: %d\n', sum(abs(t_masked_s(:))>3, 'omitnan'))
% 
% %% Compare smoothed vs unsmoothed
% fprintf('\n--- Comparison ---\n')
% fprintf('Unsmoothed: |t|>2: %d,  |t|>3: %d\n', ...
%     sum(abs(t_masked_z(:))>2,'omitnan'), ...
%     sum(abs(t_masked_z(:))>3,'omitnan'))
% fprintf('Smoothed:   |t|>2: %d,  |t|>3: %d\n', ...
%     sum(abs(t_masked_s(:))>2,'omitnan'), ...
%     sum(abs(t_masked_s(:))>3,'omitnan'))

%% Downsample to 2x2x2 supervoxels
fprintf('Creating 2x2x2 supervoxels...\n')

% New dimensions
nx2 = floor(nx/2);
ny2 = floor(ny/2);
nz2 = floor(nz/2);

V_ds = zeros(nx2, ny2, nz2, nt);
for i = 1:nx2
    for j = 1:ny2
        for k = 1:nz2
            % Average 2x2x2 block
            xi = (2*i-1):(2*i);
            yi = (2*j-1):(2*j);
            zi = (2*k-1):(2*k);
            block = V(xi, yi, zi, :);
            V_ds(i,j,k,:) = mean(block(:,:,:,:), [1 2 3]);
        end
    end
end
fprintf('Supervoxel dimensions: %d x %d x %d\n', nx2, ny2, nz2)

% Brain mask for downsampled data
mean_vol_ds = mean(V_ds, 4);
brain_mask_ds = mean_vol_ds > mean(mean_vol_ds(:)) * 0.3;
fprintf('Supervoxels in brain: %d\n', sum(brain_mask_ds(:)))

% Rerun GLM
data_ds = double(reshape(V_ds, [], nt)');
vox_mean_ds = mean(data_ds, 1);
vox_std_ds  = std(data_ds, 0, 1);
vox_std_ds(vox_std_ds < 1e-10) = 1;
data_ds_z = (data_ds - vox_mean_ds) ./ vox_std_ds;

betas_ds    = X \ data_ds_z;
residuals_ds = data_ds_z - X * betas_ds;
sigma2_ds   = sum(residuals_ds.^2) / df;
var_beta_ds = (contrast * inv(X'*X) * contrast') .* sigma2_ds;
t_map_ds    = (contrast * betas_ds) ./ sqrt(var_beta_ds);

t_map_3d_ds = reshape(t_map_ds, [nx2, ny2, nz2]);
t_masked_ds = t_map_3d_ds;
t_masked_ds(~brain_mask_ds) = NaN;

fprintf('Supervoxel GLM: |t|>2: %d,  |t|>3: %d\n', ...
    sum(abs(t_masked_ds(:))>2,'omitnan'), ...
    sum(abs(t_masked_ds(:))>3,'omitnan'))
fprintf('t-map range: %.3f to %.3f\n', ...
    min(t_map_3d_ds(:)), max(t_map_3d_ds(:)))

%% Find best supervoxel slice
n_sig_ds = zeros(nz2,1);
for z = 5:nz2-5
    n_sig_ds(z) = sum(abs(t_masked_ds(:,:,z))>2,'all','omitnan');
end
[~, best_z_ds] = max(n_sig_ds);
fprintf('Best supervoxel slice: z=%d\n', best_z_ds)

%% Plot supervoxel GLM
figure('Position', [100 100 1200 400], 'Color', 'w');

subplot(1,3,1)
imagesc(rot90(mean_vol_ds(:,:,best_z_ds)));
colormap(gca, 'gray'); colorbar; axis image;
title(sprintf('Mean brain (z=%d)', best_z_ds));
xlabel('x'); ylabel('y');

subplot(1,3,2)
imagesc(rot90(t_masked_ds(:,:,best_z_ds)));
colormap(gca, 'jet'); colorbar; axis image;
clim([-5 5]); set(gca, 'Color', 'k');
title('T-statistic map (supervoxels)');
xlabel('x'); ylabel('y');

subplot(1,3,3)
thresh_ds = t_masked_ds(:,:,best_z_ds);
thresh_ds(abs(thresh_ds) < 2) = NaN;
brain_alpha = ~isnan(rot90(thresh_ds));
imagesc(rot90(thresh_ds), 'AlphaData', brain_alpha);
colormap(gca, 'jet'); colorbar; axis image;
clim([-5 5]); set(gca, 'Color', 'k');
title('Thresholded |t|>2 (supervoxels)');
xlabel('x'); ylabel('y');

sgtitle('GLM: Pupil → BOLD (2x2x2 supervoxels, HRF convolved)');

%% Compare all approaches
% fprintf('\n=== Final Comparison ===\n')
% fprintf('Unsmoothed voxels:     |t|>2: %d,  |t|>3: %d\n', ...
%     sum(abs(t_masked_z(:))>2,'omitnan'), ...
%     sum(abs(t_masked_z(:))>3,'omitnan'))
% fprintf('3x3x3 smoothed:        |t|>2: %d,  |t|>3: %d\n', ...
%     sum(abs(t_masked_s(:))>2,'omitnan'), ...
%     sum(abs(t_masked_s(:))>3,'omitnan'))
% fprintf('2x2x2 supervoxels:     |t|>2: %d,  |t|>3: %d\n', ...
%     sum(abs(t_masked_ds(:))>2,'omitnan'), ...
%     sum(abs(t_masked_ds(:))>3,'omitnan'))


%% Clean overlay - two axes approach
figure('Position', [100 100 500 700], 'Color', 'w');

% Background anatomy
bg    = rot90(mean_vol_ds(:,:,best_z_ds));
bg_norm = (bg - min(bg(:))) / (max(bg(:)) - min(bg(:)));

% Foreground t-map
fg    = rot90(t_masked_ds(:,:,best_z_ds));
fg(abs(fg) < 2) = NaN;

% Plot grayscale brain as background
ax1 = axes('Position', [0.1 0.1 0.75 0.8]);
imagesc(bg_norm); colormap(ax1, gray);
axis image off; hold on;

% Plot t-map overlay on same axes
ax2 = axes('Position', [0.1 0.1 0.75 0.8]);
imagesc(fg, 'AlphaData', ~isnan(fg) * 0.8);
colormap(ax2, jet);
clim([-5 5]);
axis image off;
ax2.Color = 'none';  % transparent background

% Colorbar
cb = colorbar(ax2, 'Position', [0.87 0.1 0.04 0.8]);
cb.Color = 'k';
cb.Label.String = 't-statistic';
cb.Label.Color  = 'k';

title(ax2, sprintf('Pupil-BOLD GLM (z=%d, 2x2x2 supervoxels, |t|>2)', ...
    best_z_ds), 'Color', 'k', 'FontSize', 11);





%% Brain state binning based on pupil size

% Step 1: Label each timepoint as ON or OFF based on pupil
pupil_median = median(pupil_clean);
on_idx  = pupil_clean >= pupil_median;  % high pupil = ON state
off_idx = pupil_clean <  pupil_median;  % low pupil  = OFF state

fprintf('ON timepoints:  %d\n', sum(on_idx))
fprintf('OFF timepoints: %d\n', sum(off_idx))

% Step 2: Average BOLD across ON and OFF timepoints per voxel
data_on  = mean(data_2d_z(on_idx,  :), 1);  % (1 x voxels)
data_off = mean(data_2d_z(off_idx, :), 1);  % (1 x voxels)

% Step 3: Difference map (ON - OFF)
diff_map = reshape(data_on - data_off, [nx, ny, nz]);
diff_map(~brain_mask) = NaN;

fprintf('ON-OFF diff range: %.4f to %.4f\n', ...
    min(diff_map(:),[],'omitnan'), max(diff_map(:),[],'omitnan'))

% Step 4: Simple t-test at each voxel between ON and OFF
bold_on  = data_2d_z(on_idx,  :);   % (n_on  x voxels)
bold_off = data_2d_z(off_idx, :);   % (n_off x voxels)

n1 = sum(on_idx);
n2 = sum(off_idx);
mean1 = mean(bold_on,  1);
mean2 = mean(bold_off, 1);
var1  = var(bold_on,  0, 1);
var2  = var(bold_off, 0, 1);

% Two-sample t-test per voxel
t_onoff = (mean1 - mean2) ./ sqrt(var1/n1 + var2/n2);
t_onoff_3d = reshape(t_onoff, [nx, ny, nz]);
t_onoff_3d(~brain_mask) = NaN;

fprintf('ON vs OFF t-map range: %.3f to %.3f\n', ...
    min(t_onoff_3d(:),[],'omitnan'), max(t_onoff_3d(:),[],'omitnan'))
fprintf('Voxels |t|>2: %d\n', sum(abs(t_onoff_3d(:))>2,'omitnan'))

%% Step 5: Visualize - single plot (ON vs OFF timeseries)
figure('Position', [100 100 700 300], 'Color', 'w');

ax = axes('Position', [0.12 0.15 0.85 0.75]);
t_axis = 1:numel(mean_bold_clean);
plot(t_axis, mean_bold_clean, 'k-', 'LineWidth', 0.9); hold on;
scatter(find(on_idx),  mean_bold_clean(on_idx),  30, [1 0 0], 'filled');
scatter(find(off_idx), mean_bold_clean(off_idx), 30, [0 0 1], 'filled');
xlabel('Timepoint', 'Color','k');
ylabel('BOLD (z)',  'Color','k');
title('Global BOLD: ON (red) vs OFF (blue)', 'Color','k');
set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'Box', 'off');
legend({'BOLD','ON','OFF'}, 'TextColor','k', 'Color','w', 'Location','northeast');

%% Step 5: Visualize with Cyan Shading for ON State Phases
figure('Position', [100 100 850 350], 'Color', 'w');
ax = axes('Position', [0.1 0.15 0.85 0.75]);
t_axis = 1:numel(mean_bold_clean);

% --- 1. Identify and Plot Cyan Shading for ON phases ---
% Find the start and end indices of consecutive ON timepoints
on_transitions = diff([0; on_idx(:); 0]);
starts = find(on_transitions == 1);
ends   = find(on_transitions == -1) - 1;

hold on;
y_range = [-3.5 3.5]; % Setting range to cover z-score fluctuations
for i = 1:length(starts)
    % Create a rectangle (patch) from start to end of the ON phase
    patch([starts(i)-0.5, ends(i)+0.5, ends(i)+0.5, starts(i)-0.5], ...
          [y_range(1), y_range(1), y_range(2), y_range(2)], ...
          [0.85 1 1], 'EdgeColor', 'none', 'FaceAlpha', 0.7);
end

% --- 2. Plot BOLD Signal and Dots ---
% Plot the main BOLD line on top of the shading
plot(t_axis, mean_bold_clean, 'k-', 'LineWidth', 1.2); 

% --- 3. Formatting ---
xlabel('Timepoint (TR = 1s)');
ylabel('Global BOLD (z)');
title('Global BOLD: Pupil-ON/OFF Phases');
ylim(y_range);
xlim([1 205]);
set(ax, 'Box', 'off', 'Layer', 'top'); % 'Layer top' ensures the lines stay visible over shading

% Update legend to include the shading info
h_on_patch = patch(NaN, NaN, [0.85 1 1], 'EdgeColor', 'none'); % Dummy for legend
legend([h_on_patch, plot(NaN,NaN,'k-')], {'ON State Duration', 'Global BOLD'}, ...
       'Location', 'northeast');


%% Stricter threshold + mean BOLD comparison
fprintf('\nMean BOLD in ON state:  %.4f\n', mean(mean_bold_clean(on_idx)))
fprintf('Mean BOLD in OFF state: %.4f\n', mean(mean_bold_clean(off_idx)))
fprintf('Difference: %.4f\n', ...
    mean(mean_bold_clean(on_idx)) - mean(mean_bold_clean(off_idx)))

%% Replot with lower threshold |t|>2
figure('Position', [100 100 500 600], 'Color', 'k');

bg3 = rot90(mean_vol(:,:,best_z_onoff));
bg3 = (bg3 - min(bg3(:))) / (max(bg3(:)) - min(bg3(:)));

ax1 = axes('Position', [0.1 0.15 0.7 0.75]);
imagesc(bg3); colormap(ax1, gray); axis image off;

fg3 = rot90(t_onoff_3d(:,:,best_z_onoff));
fg3(abs(fg3) < 2.5) = NaN;  % lowered to |t|>2.5

ax2 = axes('Position', [0.1 0.15 0.7 0.75]);
imagesc(fg3, 'AlphaData', ~isnan(fg3) * 0.85);
colormap(ax2, jet); clim([-5 5]);
axis image off; ax2.Color = 'none';

cb = colorbar(ax2, 'Position', [0.83 0.15 0.04 0.75]);
cb.Color = 'w'; cb.Label.String = 't-statistic';
cb.Label.Color = 'w'; cb.FontSize = 10;

title(ax2, sprintf(['ON vs OFF Brain State\n' ...
    'Pupil-binned, z=%d, |t|>2.5 (p<0.05 uncorrected)'], best_z_onoff), ...
    'Color', 'w', 'FontSize', 11);

% Count
fprintf('Voxels surviving |t|>2.5: %d\n', ...
    sum(abs(t_onoff_3d(:,:,best_z_onoff))>2.5,'all','omitnan'))
%% Bar chart comparing ON vs OFF global BOLD
figure('Position', [100 100 400 350], 'Color', 'w');
means = [mean(mean_bold_clean(on_idx)), mean(mean_bold_clean(off_idx))];
sems  = [std(mean_bold_clean(on_idx))/sqrt(sum(on_idx)), ...
         std(mean_bold_clean(off_idx))/sqrt(sum(off_idx))];

bar(means, 'FaceColor', 'flat', 'CData', [1 0.3 0.3; 0.3 0.3 1], ...
    'EdgeColor', 'none'); hold on;
errorbar(1:2, means, sems, 'k.', 'LineWidth', 1.5);
xticks([1 2]); xticklabels({'ON state', 'OFF state'});
ylabel('Mean BOLD (z)');
title('Global BOLD: ON vs OFF Brain State');
box off;



%% RAW vs PREPROCESSED FIGURES

% Figure A: Raw fMRI pipeline - using middle timepoint
figure('Position', [100 100 1000 350], 'Color', 'w');

subplot(1,3,1)
imagesc(rot90(squeeze(V(:,:,best_z_onoff,mid_t))));
colormap(gca, 'gray'); axis image off; colorbar;
title(sprintf('Raw fMRI (t=%d, z=%d)', mid_t, best_z_onoff));

subplot(1,3,2)
imagesc(rot90(mean_vol(:,:,best_z_onoff)));
colormap(gca, 'gray'); axis image off; colorbar;
title('Mean brain (avg 205 TRs)');

subplot(1,3,3)
imagesc(rot90(double(brain_mask(:,:,best_z_onoff))));
colormap(gca, 'gray'); axis image off; colorbar;
title('Brain mask (30% threshold)');

sgtitle('fMRI Preprocessing: Raw → Mean → Brain Mask');
%% Figure B: Raw vs cleaned BOLD timeseries
figure('Position', [100 100 900 500], 'Color', 'w');

subplot(2,1,1)
plot(mean_bold, 'b', 'LineWidth', 1);
xlabel('Timepoint (TR)'); ylabel('Signal (a.u.)');
title('Raw global mean BOLD (unprocessed)');
xlim([1 205]); box off;

subplot(2,1,2)
plot(mean_bold_clean, 'b', 'LineWidth', 1); hold on;
plot(pupil_clean, 'r', 'LineWidth', 1);
legend('BOLD (z-scored, cleaned)', 'Pupil (z-scored)', ...
    'Location', 'northeast');
xlabel('Timepoint (TR = 1s)'); ylabel('Z-score');
title('Preprocessed: z-scored + spike removed');
xlim([1 205]); box off;

sgtitle('BOLD Signal: Raw vs Preprocessed');

%% Figure C: Raw vs cleaned pupil
figure('Position', [100 100 900 500], 'Color', 'w');

subplot(2,1,1)
plot(pupil, 'r', 'LineWidth', 1);
xlabel('Timepoint'); ylabel('Pupil size (a.u.)');
title('Raw pupil timeseries');
xlim([1 205]); box off;

subplot(2,1,2)
plot(pupil_clean, 'r', 'LineWidth', 1);
xlabel('Timepoint (TR = 1s)'); ylabel('Z-score');
title('Preprocessed pupil (z-scored)');
xlim([1 205]); box off;

sgtitle('Pupil Signal: Raw vs Preprocessed');

%% Figure D: GLM design matrix + HRF
figure('Position', [100 100 900 400], 'Color', 'w');

subplot(1,3,1)
hrf_plot = spm_hrf(1);
plot(hrf_plot, 'k-', 'LineWidth', 2);
xlabel('Time (s)'); ylabel('Amplitude');
title('Canonical HRF');
box off;

subplot(1,3,2)
plot(pupil_clean, 'r', 'LineWidth', 1); hold on;
plot(pupil_reg,   'k', 'LineWidth', 1.5);
legend('Raw pupil (z)', 'HRF convolved', 'Location', 'northeast');
xlabel('Timepoint'); ylabel('Z-score');
title('Pupil regressor: before/after HRF');
xlim([1 205]); box off;

subplot(1,3,3)
imagesc(X); colormap(gca, 'gray'); colorbar;
title('GLM Design Matrix');
xlabel('Regressors'); ylabel('Timepoints');
xticks([1 2]); xticklabels({'Pupil (HRF)', 'Intercept'});

sgtitle('GLM Pipeline: HRF → Regressor → Design Matrix');

%% 

% Find coordinates of strongest activation
[~, top_idx] = sort(abs(t_onoff_3d(:)), 'descend', ...
    'MissingPlacement', 'last');
top_idx = top_idx(1:20);
[px, py, pz] = ind2sub([nx, ny, nz], top_idx);
fprintf('Top activation coordinates:\n')
disp([px, py, pz])

%% Updated ROIs based on actual activation coordinates
ROIs = struct();

ROIs(1).name = 'Dorsal Cortex (Motor/Visual)';
ROIs(1).center = [70, 72, 55];   % cluster center of dorsal group
ROIs(1).radius = 8;

ROIs(2).name = 'Thalamus';
ROIs(2).center = [85, 94, 34];   % cluster center of thalamic group
ROIs(2).radius = 6;

ROIs(3).name = 'Brainstem (LC/PRN area)';
ROIs(3).center = [18, 78, 15];   % cluster center of brainstem group
ROIs(3).radius = 6;

ROIs(4).name = 'Anterior Cortex';
ROIs(4).center = [91, 55, 63];   % dorsal anterior cluster
ROIs(4).radius = 5;

ROIs(5).name = 'Posterior Thalamus';
ROIs(5).center = [74, 120, 23];  % posterior subcortical
ROIs(5).radius = 5;

%% Extract ROI timeseries and t-values
[X_grid, Y_grid, Z_grid] = ndgrid(1:nx, 1:ny, 1:nz);

fprintf('\n=== ROI Results ===\n')
fprintf('%-30s | Voxels | GLM t | ON-OFF t | Mean BOLD ON | Mean BOLD OFF\n', 'Region')
fprintf('%s\n', repmat('-',1,85))

for r = 1:length(ROIs)
    cx = ROIs(r).center(1);
    cy = ROIs(r).center(2);
    cz = ROIs(r).center(3);
    rad = ROIs(r).radius;
    
    sphere = sqrt((X_grid-cx).^2 + (Y_grid-cy).^2 + ...
                  (Z_grid-cz).^2) <= rad;
    sphere = sphere & brain_mask;
    
    % GLM t-value
    t_glm_roi = mean(t_map_3d_z(sphere), 'omitnan');
    
    % ON vs OFF t-value
    t_onoff_roi = mean(t_onoff_3d(sphere), 'omitnan');
    
    % Mean BOLD during ON and OFF states
    roi_ts = data_2d_z(:, sphere(:));  % (205 x n_voxels)
    bold_on_roi  = mean(roi_ts(on_idx,  :), 'all');
    bold_off_roi = mean(roi_ts(off_idx, :), 'all');
    
    fprintf('%-30s | %6d | %5.3f | %8.3f | %12.4f | %12.4f\n', ...
        ROIs(r).name, sum(sphere(:)), t_glm_roi, ...
        t_onoff_roi, bold_on_roi, bold_off_roi)
end


%% Tighter ROIs centered on actual peak voxels
clear ROIs

% Use your actual top coordinates directly
peak_coords = [
    94, 71, 53;   % Dorsal cortex right
    43, 71, 58;   % Dorsal cortex left
    91, 50, 63;   % Anterior cortex right
    44, 89, 64;   % Anterior cortex left
    90, 72, 37;   % Thalamus right
    74, 117, 34;  % Posterior thalamus
    19, 123, 29;  % Subcortical posterior
    14, 89, 7;    % Brainstem left
    95, 66, 14;   % Brainstem right
    18, 78, 20;   % Brainstem anterior
];

peak_names = {
    'Dorsal Cortex R';
    'Dorsal Cortex L';
    'Anterior Cortex R';
    'Anterior Cortex L';
    'Thalamus R';
    'Posterior Thalamus';
    'Subcortical Posterior';
    'Brainstem L';
    'Brainstem R';
    'Brainstem Anterior';
};

fprintf('\n=== Peak Voxel ROI Results ===\n')
fprintf('%-25s | GLM t | ON-OFF t | BOLD ON  | BOLD OFF\n', 'Region')
fprintf('%s\n', repmat('-',1,65))

for r = 1:size(peak_coords,1)
    cx = peak_coords(r,1);
    cy = peak_coords(r,2);
    cz = peak_coords(r,3);

    % Small 3-voxel radius sphere
    sphere = sqrt((X_grid-cx).^2 + (Y_grid-cy).^2 + ...
                  (Z_grid-cz).^2) <= 3;
    sphere = sphere & brain_mask;

    if sum(sphere(:)) == 0
        fprintf('%-25s | outside brain mask\n', peak_names{r})
        continue
    end

    t_glm_roi   = mean(t_map_3d_z(sphere), 'omitnan');
    t_onoff_roi = mean(t_onoff_3d(sphere),  'omitnan');
    roi_ts      = data_2d_z(:, sphere(:));
    bold_on_roi  = mean(roi_ts(on_idx,  :), 'all');
    bold_off_roi = mean(roi_ts(off_idx, :), 'all');

    fprintf('%-25s | %5.3f | %8.3f | %8.4f | %8.4f\n', ...
        peak_names{r}, t_glm_roi, t_onoff_roi, ...
        bold_on_roi, bold_off_roi)
end

%% Summary bar chart of ROI ON vs OFF BOLD
figure('Position', [100 100 900 400], 'Color', 'w');

region_names = {'Ant Ctx R', 'Brainstem L', 'Brainstem R', ...
    'Post Thal', 'Sub Post', 'Brainstem Ant', ...
    'Dors Ctx R', 'Post Thal', 'Ant Ctx L', ...
    'Thal R', 'Dors Ctx L'};

bold_on_vals  = [0.0477, 0.0433, 0.0189, 0.0134, 0.0118, 0.0074, ...
                  0.0013, -0.0262, -0.0491, -0.0536];
bold_off_vals = [-0.0482, -0.0437, -0.0191, -0.0136, -0.0119, -0.0075, ...
                  -0.0013, 0.0264, 0.0496, 0.0542];

names_sorted = {'Ant Ctx R', 'Brainstem L', 'Brainstem R', ...
                'Post Thal', 'Sub Post', 'Brainstem Ant', ...
                'Dors Ctx R', 'Ant Ctx L', 'Thal R', 'Dors Ctx L'};

x = 1:length(bold_on_vals);
bar_data = [bold_on_vals; bold_off_vals]';

b = bar(x, bar_data, 'grouped');
b(1).FaceColor = [0.9 0.3 0.3];  % red = ON
b(2).FaceColor = [0.3 0.3 0.9];  % blue = OFF

yline(0, '-k', 'LineWidth', 1);
xticks(x);
xticklabels(names_sorted);
xtickangle(30);
ylabel('Mean BOLD (z)');
title('Mean BOLD Signal: ON vs OFF Brain State by Region');
legend('ON state (large pupil)', 'OFF state (small pupil)', ...
    'Location', 'northeast');
box off;


%% Find true peak positive and negative correlation voxels
% Use r_map_smooth which was computed directly from pupil-BOLD correlation

%Recompute voxelwise correlation map
pupil_lagged    = pupil_clean(1:end-4);         % 201 timepoints
bold_trimmed    = data_2d_z(5:end, :);          % (201 x voxels)

% Z-score pupil lagged
pupil_z_lag = (pupil_lagged - mean(pupil_lagged)) / std(pupil_lagged);

% Pearson r per voxel
r_voxels = (bold_trimmed' * pupil_z_lag) / (length(pupil_lagged) - 1);

% Map back to 3D
r_map = zeros(nx, ny, nz);
r_map(:) = r_voxels;
r_map(~brain_mask) = NaN;

% Smooth within brain mask
r_map_smooth = zeros(nx, ny, nz);
h = fspecial('gaussian', [3 3], 1);
for z = 1:nz
    slice = r_map(:,:,z);
    mask_slice = brain_mask(:,:,z);
    slice_smooth = imfilter(slice, h, 'replicate');
    slice_smooth(~mask_slice) = 0;
    r_map_smooth(:,:,z) = slice_smooth;
end

fprintf('r_map_smooth range: %.4f to %.4f\n', ...
    min(r_map_smooth(:),[],'omitnan'), ...
    max(r_map_smooth(:),[],'omitnan'))

% Now find true peak voxels
r_brain = r_map_smooth;
r_brain(~brain_mask) = NaN;

[max_r, max_idx] = max(r_brain(:));
[px_pos, py_pos, pz_pos] = ind2sub([nx,ny,nz], max_idx);
fprintf('Peak POSITIVE r = %.4f at voxel (%d,%d,%d)\n', ...
    max_r, px_pos, py_pos, pz_pos)

[min_r, min_idx] = min(r_brain(:));
[px_neg, py_neg, pz_neg] = ind2sub([nx,ny,nz], min_idx);
fprintf('Peak NEGATIVE r = %.4f at voxel (%d,%d,%d)\n', ...
    min_r, px_neg, py_neg, pz_neg)

% Extract timeseries from true peak voxels
pos_sphere3 = sqrt((X_grid-px_pos).^2 + (Y_grid-py_pos).^2 + ...
                   (Z_grid-pz_pos).^2) <= 3 & brain_mask;
neg_sphere3 = sqrt((X_grid-px_neg).^2 + (Y_grid-py_neg).^2 + ...
                   (Z_grid-pz_neg).^2) <= 3 & brain_mask;

pos_ts3 = mean(data_2d_z(:, pos_sphere3(:)), 2);
neg_ts3 = mean(data_2d_z(:, neg_sphere3(:)), 2);

C_pos3 = corrcoef(pos_ts3(5:end), pupil_clean(1:end-4));
C_neg3 = corrcoef(neg_ts3(5:end), pupil_clean(1:end-4));
r_pos3 = C_pos3(1,2);
r_neg3 = C_neg3(1,2);
fprintf('Verified positive r = %.4f\n', r_pos3)
fprintf('Verified negative r = %.4f\n', r_neg3)
% Find top positive correlation voxel
r_brain = r_map_smooth;
r_brain(~brain_mask) = NaN;

[max_r, max_idx] = max(r_brain(:));
[px_pos, py_pos, pz_pos] = ind2sub([nx,ny,nz], max_idx);
fprintf('Peak POSITIVE r = %.4f at voxel (%d,%d,%d)\n', ...
    max_r, px_pos, py_pos, pz_pos)

% Find top negative correlation voxel
[min_r, min_idx] = min(r_brain(:));
[px_neg, py_neg, pz_neg] = ind2sub([nx,ny,nz], min_idx);
fprintf('Peak NEGATIVE r = %.4f at voxel (%d,%d,%d)\n', ...
    min_r, px_neg, py_neg, pz_neg)

% Extract timeseries from true peak voxels
pos_sphere3 = sqrt((X_grid-px_pos).^2 + (Y_grid-py_pos).^2 + ...
                   (Z_grid-pz_pos).^2) <= 3 & brain_mask;
neg_sphere3 = sqrt((X_grid-px_neg).^2 + (Y_grid-py_neg).^2 + ...
                   (Z_grid-pz_neg).^2) <= 3 & brain_mask;

pos_ts3 = mean(data_2d_z(:, pos_sphere3(:)), 2);
neg_ts3 = mean(data_2d_z(:, neg_sphere3(:)), 2);

C_pos3 = corrcoef(pos_ts3(5:end), pupil_clean(1:end-4));
C_neg3 = corrcoef(neg_ts3(5:end), pupil_clean(1:end-4));
r_pos3 = C_pos3(1,2);
r_neg3 = C_neg3(1,2);
fprintf('Verified positive r = %.4f\n', r_pos3)
fprintf('Verified negative r = %.4f\n', r_neg3)

%% Final positive vs negative ROI comparison figure
t_axis = 1:205;
x_fit  = linspace(min(pupil_clean), max(pupil_clean), 100);

figure('Position', [100 100 1100 800], 'Color', 'w');

% Panel 1: Positive ROI timeseries
subplot(2,2,1)
plot(t_axis, pos_ts3, 'r-', 'LineWidth', 1.2); hold on;
plot(t_axis, pupil_clean, 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8);
legend('BOLD (z)', 'Pupil (z)', 'Location', 'northwest');
xlabel('Timepoint (TR = 1s)'); ylabel('Z-score');
title(sprintf('Positive ROI (%d,%d,%d) — Brainstem/Ventral\nr = %.3f at lag +4s', ...
    px_pos, py_pos, pz_pos, r_pos3));
xlim([1 205]); box off;

% Panel 2: Negative ROI timeseries
subplot(2,2,2)
plot(t_axis, neg_ts3, 'b-', 'LineWidth', 1.2); hold on;
plot(t_axis, pupil_clean, 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8);
legend('BOLD (z)', 'Pupil (z)', 'Location', 'northwest');
xlabel('Timepoint (TR = 1s)'); ylabel('Z-score');
title(sprintf('Negative ROI (%d,%d,%d) — Dorsal Region\nr = %.3f at lag +4s', ...
    px_neg, py_neg, pz_neg, r_neg3));
xlim([1 205]); box off;

% Panel 3: Scatter positive
subplot(2,2,3)
scatter(pupil_clean(1:end-4), pos_ts3(5:end), ...
    20, 'r', 'filled', 'MarkerFaceAlpha', 0.4); hold on;
p1 = polyfit(pupil_clean(1:end-4), pos_ts3(5:end), 1);
plot(x_fit, polyval(p1, x_fit), 'r-', 'LineWidth', 2.5);
xline(0, '--k', 'Alpha', 0.3);
yline(0, '--k', 'Alpha', 0.3);
xlabel('Pupil size (z, lag 0)');
ylabel('BOLD (z, lag +4s)');
title(sprintf('Positive ROI\nr = %.3f', r_pos3));
box off;

% Panel 4: Scatter negative
subplot(2,2,4)
scatter(pupil_clean(1:end-4), neg_ts3(5:end), ...
    20, 'b', 'filled', 'MarkerFaceAlpha', 0.4); hold on;
p2 = polyfit(pupil_clean(1:end-4), neg_ts3(5:end), 1);
plot(x_fit, polyval(p2, x_fit), 'b-', 'LineWidth', 2.5);
xline(0, '--k', 'Alpha', 0.3);
yline(0, '--k', 'Alpha', 0.3);
xlabel('Pupil size (z, lag 0)');
ylabel('BOLD (z, lag +4s)');
title(sprintf('Negative ROI\nr = %.3f', r_neg3));
box off;

sgtitle('Positive vs Negative Pupil-BOLD Coupling: Peak Voxel ROIs', ...
    'FontSize', 13, 'FontWeight', 'bold');

%% Fix: colorbar on right panel is cut off + center both panels better
figure('Position', [100 100 1100 550], 'Color', 'w');

% Left - ON>OFF
bg_on  = rot90(mean_vol(:,:,best_z_on));
bg_on  = (bg_on - min(bg_on(:))) / (max(bg_on(:)) - min(bg_on(:)));
ax1 = axes('Position', [0.04 0.08 0.41 0.80]);
imagesc(bg_on); colormap(ax1, gray); axis image off;
fg_on = rot90(t_onoff_3d(:,:,best_z_on));
fg_on(fg_on < 2.5) = NaN;
ax2 = axes('Position', [0.04 0.08 0.41 0.80]);
imagesc(fg_on, 'AlphaData', ~isnan(fg_on) * 0.85);
colormap(ax2, hot); clim([2.5 5]);
axis image off; ax2.Color = 'none';
cb1 = colorbar(ax2, 'Position', [0.46 0.08 0.02 0.80]);
cb1.Color = 'k'; cb1.FontSize = 9;
cb1.Label.String = 't-statistic'; cb1.Label.Color = 'k';
title(ax2, sprintf('ON > OFF  (High Arousal)\nz=%d  |  n=%d voxels  |  t > 2.5', ...
    best_z_on, n_on_per_slice(best_z_on)), ...
    'Color', 'k', 'FontSize', 11, 'FontWeight', 'bold');

% Right - OFF>ON
bg_off = rot90(mean_vol(:,:,best_z_off));
bg_off = (bg_off - min(bg_off(:))) / (max(bg_off(:)) - min(bg_off(:)));
ax3 = axes('Position', [0.52 0.08 0.41 0.80]);
imagesc(bg_off); colormap(ax3, gray); axis image off;
fg_off = rot90(t_onoff_3d(:,:,best_z_off));
fg_off(fg_off > -2.5) = NaN;
fg_off = -fg_off;
ax4 = axes('Position', [0.52 0.08 0.41 0.80]);
imagesc(fg_off, 'AlphaData', ~isnan(fg_off) * 0.85);
colormap(ax4, winter); clim([2.5 5]);
axis image off; ax4.Color = 'none';
cb2 = colorbar(ax4, 'Position', [0.94 0.08 0.02 0.80]);
cb2.Color = 'k'; cb2.FontSize = 9;
cb2.Label.String = '|t-statistic|'; cb2.Label.Color = 'k';
title(ax4, sprintf('OFF > ON  (Low Arousal)\nz=%d  |  n=%d voxels  |  t < -2.5', ...
    best_z_off, n_off_per_slice(best_z_off)), ...
    'Color', 'k', 'FontSize', 11, 'FontWeight', 'bold');

%% SECTION: ROI-Specific BOLD-Pupil Comparison
% Coordinates provided: 
% Positive ROI: (102, 105, 18)
% Negative ROI: (88, 98, 21)

rois = { [102, 105, 18], [88, 98, 21] };
roi_names = {'Positive ROI', 'Negative ROI'};

for i = 1:length(rois)
    coord = rois{i};
    
    % 1. Extract and Z-score the ROI time series
    % We use V_smooth for better signal-to-noise ratio
    roi_raw = squeeze(V_smooth(coord(1), coord(2), coord(3), :));
    roi_z   = (roi_raw - mean(roi_raw)) / std(roi_raw);
    
    % 2. Calculate ROI-specific lag correlation
    roi_r_lags = zeros(size(lags));
    for j = 1:length(lags)
        lag = lags(j);
        if lag > 0
            C = corrcoef(roi_z(lag+1:end), pupil_clean(1:end-lag));
        elseif lag < 0
            C = corrcoef(roi_z(1:end+lag), pupil_clean(-lag+1:end));
        else
            C = corrcoef(roi_z, pupil_clean);
        end
        roi_r_lags(j) = C(1,2);
    end
    
    % Find best lag and r-value
    [max_r, max_idx] = max(abs(roi_r_lags));
    best_r = roi_r_lags(max_idx);
    best_l = lags(max_idx);
    
    % 3. Generate the Figure
    figure('Position', [100 100 900 350], 'Color', 'w');
    hold on;
    
    % --- Add Cyan Shading for ON states ---
    y_limits = [-3.5 3.5];
    for s = 1:length(starts)
        patch([starts(s)-0.5, ends(s)+0.5, ends(s)+0.5, starts(s)-0.5], ...
              [y_limits(1), y_limits(1), y_limits(2), y_limits(2)], ...
              [0.85 1 1], 'EdgeColor', 'none', 'FaceAlpha', 0.6);
    end
    
    % --- Plot Signals ---
    p1 = plot(roi_z, 'b', 'LineWidth', 1.3); 
    p2 = plot(pupil_clean, 'r', 'LineWidth', 1.3);
    
    % Create a dummy patch for the legend
    h_patch = patch(NaN, NaN, [0.85 1 1], 'EdgeColor', 'none');
    
    % Formatting
    legend([p1, p2, h_patch], ...
        {sprintf('%s (z)', roi_names{i}), 'Pupil (z)', 'Pupil ON State'}, ...
        'Location', 'northeast');
    
    xlabel('Timepoint (TR = 1s)'); ylabel('Z-score');
    title(sprintf('%s vs Pupil (r = %.3f, lag = %d TR)', ...
          roi_names{i}, best_r, best_l));
    
    xlim([1 205]); ylim(y_limits); 
    set(gca, 'Layer', 'top', 'Box', 'off'); % Keeps signal lines on top of shading
    grid off;
end



