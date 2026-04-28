%% Multimodal Integration: ACh, NE, Pupil, and fMRI
% Goal 1: Do signals reflect the same brain state?
% Goal 2: Is fMRI more tightly coupled to global state than NT recordings?

%% 0. LOAD DATA
load('/Users/sowinglum/Downloads/neurophysiology/fmri/fMRI.mat');       % → variable V: [102 x 197 x 75 x 205]
load('/Users/sowinglum/Downloads/neurophysiology/fmri/pupil 1.mat');    % → check variable name with: whos

% Check what pupil_1.mat contains — run this once then comment out
disp('=== Variables in pupil 1.mat ==='); whos

ACh_raw = readmatrix('/Users/sowinglum/Downloads/neurophysiology/Ach/pupil_ACh.csv');  % [54372 x 3]: time, pupil, ACh
NE_raw  = readmatrix('/Users/sowinglum/Downloads/neurophysiology/NE/pupil_NE.csv');   % [55007 x 3]: time, pupil, NE

% Correct column order: [time, pupil, NT]
ach_time   = ACh_raw(:,1);
ach_pupil  = ACh_raw(:,2);
ach_signal = ACh_raw(:,3);

ne_time    = NE_raw(:,1);
ne_pupil   = NE_raw(:,2);
ne_signal  = NE_raw(:,3);

%% 1. PRE-PROCESS

TR   = 1;    % ← confirm your TR in seconds
nT   = 205;    % fMRI timepoints
t_fmri = (0:nT-1)' * TR;

% 1.1 Reshape 4D fMRI → [Time x Voxels]
[x, y, z, t] = size(V);
fMRI_2D = reshape(V, [], t)';   % [205 x nVoxels]

% Global Signal
GS = mean(fMRI_2D, 2);          % [205 x 1]

% 1.2 Low-pass filter + resample NT signals to 205 points
fs_ach = 1 / mean(diff(ach_time));
fs_ne  = 1 / mean(diff(ne_time));
cutoff = 0.1;   % Hz — keeps slow arousal fluctuations, adjust if needed

[b_ach, a_ach] = butter(4, cutoff / (fs_ach/2), 'low');
[b_ne,  a_ne]  = butter(4, cutoff / (fs_ne/2),  'low');

ach_filt = filtfilt(b_ach, a_ach, ach_signal);
ne_filt  = filtfilt(b_ne,  a_ne,  ne_signal);

% Resample to 205 points matching fMRI
ach_ds = resample(ach_filt, nT, length(ach_filt));
ne_ds  = resample(ne_filt,  nT, length(ne_filt));

% Pupil from NT session (use ACh file as reference)
pupil_nt_ds = resample(ach_pupil, nT, length(ach_pupil));

% 1.3 fMRI-session pupil
% *** Replace 'pupil' below with actual variable name from whos output ***
% Common possibilities: pupil, pupil_data, P, data
pupil_fmri_raw = pupil;        % 1x205 from pupil_1.mat
pupil_fmri = zscore(pupil_fmri_raw');   % transpose to 205x1, then z-score

% 1.4 Z-score all signals
GS          = zscore(GS);
ach_ds      = zscore(ach_ds);
ne_ds       = zscore(ne_ds);
pupil_nt_ds = zscore(pupil_nt_ds);
pupil_fmri  = zscore(pupil_fmri);   % uncomment once defined

% Detrend to remove slow drift before cross-correlation
GS     = detrend(GS);
ach_ds = detrend(ach_ds);
ne_ds  = detrend(ne_ds);

%% 2. GOAL 1: DO SIGNALS REFLECT THE SAME BRAIN STATE?

max_lag = 10;   % TRs

[xc_ach, lags] = xcorr(GS, ach_ds, max_lag, 'coeff');
[xc_ne,  ~]    = xcorr(GS, ne_ds,  max_lag, 'coeff');

[~, idx_ach] = max(abs(xc_ach));
[~, idx_ne]  = max(abs(xc_ne));
fprintf('Peak ACh-fMRI lag: %.1f s (TR %+d)\n', lags(idx_ach)*TR, lags(idx_ach));
fprintf('Peak NE-fMRI  lag: %.1f s (TR %+d)\n', lags(idx_ne)*TR,  lags(idx_ne));

% Coherence (frequency-domain coupling)
[C_ach, F] = mscohere(GS, ach_ds, [], [], [], 1/TR);
[C_ne,  ~] = mscohere(GS, ne_ds,  [], [], [], 1/TR);

% figure('Color','w','Name','Goal 1: Lag & Coherence');
% subplot(2,1,1);
% plot(lags*TR, xc_ach, 'c', lags*TR, xc_ne, 'm', 'LineWidth', 1.5);
% xline(0,'k--'); yline(0,'k:');
% legend('ACh','NE','TextColor','k','Color','none');
% title('Cross-correlation: NT vs fMRI Global Signal','Color','k');
% xlabel('Lag (s)','Color','k'); ylabel('Pearson r','Color','k');
% set(gca,'Color','w','XColor','k','YColor','k');
% 
% subplot(2,1,2);
% plot(F, C_ach, 'c', F, C_ne, 'm', 'LineWidth', 1.5);
% title('Magnitude-Squared Coherence: NT vs fMRI GS','Color','k');
% xlabel('Frequency (Hz)','Color','k'); ylabel('Coherence','Color','k');
% legend('ACh','NE','TextColor','k','Color','none');
% set(gca,'Color','w','XColor','k','YColor','k');

figure('Color','w','Name','ACh vs NE spatial fingerprints');
scatter(rho_ach, rho_ne, 1, [0.5 0.5 0.5], 'filled', ...
    'MarkerFaceAlpha', 0.1);
hold on;
xline(0,'k--'); yline(0,'k--');
xlabel('ACh-voxel r','Color','k');
ylabel('NE-voxel r','Color','k');
title('Spatial fingerprint: ACh vs NE','Color','k');
set(gca,'Color','w','XColor','k','YColor','k');

% Add correlation between the two maps
r_spatial = corr(rho_ach', rho_ne');
text(0.05, 0.92, sprintf('r = %.3f', r_spatial), ...
    'Units','normalized','Color','k','FontSize',11);




%% 3. GOAL 2: COUPLING STRENGTH — fMRI GS vs NT

% Tonic (slow) component via 10-s sliding window
win = round(10 / TR);
ach_tonic  = movmean(ach_ds, win);
ach_phasic = ach_ds - ach_tonic;
ne_tonic   = movmean(ne_ds, win);
ne_phasic  = ne_ds - ne_tonic;

% *** Uncomment block below once pupil_fmri is defined ***
r_gs_pupil  = corr(GS, pupil_fmri);
r_ach_pupil = corr(ach_ds, pupil_fmri);
r_ne_pupil  = corr(ne_ds,  pupil_fmri);
fprintf('\n--- Coupling strengths (correlation with pupil) ---\n');
fprintf('fMRI GS: r = %.3f\n', r_gs_pupil);
fprintf('ACh:     r = %.3f\n', r_ach_pupil);
fprintf('NE:      r = %.3f\n', r_ne_pupil);
%
% % Variance partitioning
mdl_nt_only = fitlm([ach_ds, ne_ds],       pupil_fmri);
mdl_full    = fitlm([ach_ds, ne_ds, GS],   pupil_fmri);
fprintf('\n--- Variance partitioning ---\n');
fprintf('R² NT only:            %.3f\n', mdl_nt_only.Rsquared.Ordinary);
fprintf('R² NT + fMRI GS:       %.3f\n', mdl_full.Rsquared.Ordinary);
fprintf('Unique fMRI GS var:    %.3f\n', mdl_full.Rsquared.Ordinary - mdl_nt_only.Rsquared.Ordinary);

%% 4. SPATIAL TOPOGRAPHY: NT vs Pupil voxelwise correlations

nV = size(fMRI_2D, 2);
rho_ach = zeros(1, nV);
rho_ne  = zeros(1, nV);

fprintf('Computing voxelwise correlations (%d voxels)...\n', nV);
for v = 1:nV
    ts = fMRI_2D(:,v);
    rho_ach(v) = corr(ach_ds, ts);
    rho_ne(v)  = corr(ne_ds,  ts);
end

% *** Uncomment once pupil_fmri defined ***
rho_pupil = zeros(1, nV);
for v = 1:nV
    rho_pupil(v) = corr(pupil_fmri, fMRI_2D(:,v));
end

% Statistical comparison of distributions
[h_an, p_an] = kstest2(rho_ach, rho_ne);
fprintf('\nKS test ACh-voxel vs NE-voxel: h=%d, p=%.4f\n', h_an, p_an);
fprintf('ACh-voxel r: mean=%.3f, std=%.3f\n', mean(rho_ach), std(rho_ach));
fprintf('NE-voxel  r: mean=%.3f, std=%.3f\n', mean(rho_ne),  std(rho_ne));

figure('Color','w','Name','Goal 2: Spatial Distributions');
histogram(rho_ach, 60, 'FaceColor','c', 'FaceAlpha',0.5, 'EdgeColor','none'); hold on;
histogram(rho_ne,  60, 'FaceColor','m', 'FaceAlpha',0.5, 'EdgeColor','none');
xline(0,'k--');
title(sprintf('Voxelwise correlations (KS p=%.4f)', p_an),'Color','k');
xlabel('Pearson r','Color','k'); ylabel('Voxel count','Color','k');
legend('ACh-Voxel','NE-Voxel','TextColor','k','Color','none');
set(gca,'Color','w','XColor','k','YColor','k');


%% 5. DATA-DRIVEN ROI: LC-like (NE) and BF-like (ACh) networks

% 5.1 Threshold: top 5% most correlated voxels for each NT
thresh_pct = 95;

ne_thresh  = prctile(rho_ne,  thresh_pct);
ach_thresh = prctile(rho_ach, thresh_pct);  % top 5% positive
% For ACh we also want most negative (global suppression signature)
ach_thresh_neg = prctile(rho_ach, 100 - thresh_pct);

% Boolean masks (in flattened voxel space)
LC_mask  = rho_ne  >= ne_thresh;           % NE-preferring voxels
BF_mask  = rho_ach <= ach_thresh_neg;      % ACh-preferring (negative BOLD)

fprintf('LC-like voxels:  %d\n', sum(LC_mask));
fprintf('BF-like voxels:  %d\n', sum(BF_mask));
fprintf('Overlap:         %d\n', sum(LC_mask & BF_mask));

% 5.2 Extract mean timecourses from each ROI
LC_tc = mean(fMRI_2D(:, LC_mask), 2);   % [205 x 1]
BF_tc = mean(fMRI_2D(:, BF_mask), 2);   % [205 x 1]

LC_tc = zscore(LC_tc);
BF_tc = zscore(BF_tc);

% 5.3 Correlate ROI timecourses with NT signals
r_LC_ne  = corr(LC_tc, ne_ds);
r_LC_ach = corr(LC_tc, ach_ds);
r_BF_ne  = corr(BF_tc, ne_ds);
r_BF_ach = corr(BF_tc, ach_ds);

fprintf('\n--- ROI-NT coupling ---\n');
fprintf('LC-like ROI ~ NE:  r = %.3f\n', r_LC_ne);
fprintf('LC-like ROI ~ ACh: r = %.3f\n', r_LC_ach);
fprintf('BF-like ROI ~ NE:  r = %.3f\n', r_BF_ne);
fprintf('BF-like ROI ~ ACh: r = %.3f\n', r_BF_ach);

% 5.4 Plot timecourses over time
figure('Color','k','Name','ROI Timecourses vs NT signals');

subplot(2,1,1);
plot(t_fmri, LC_tc, 'Color',[0.4 0.8 1], 'LineWidth',1.5); hold on;
plot(t_fmri, ne_ds, 'Color',[1 0.6 0.2], 'LineWidth',1.2);
title(sprintf('LC-like ROI vs NE  (r=%.3f)', r_LC_ne), 'Color','w');
xlabel('Time (s)','Color','w'); ylabel('Z-score','Color','w');
legend('LC-like fMRI','NE','TextColor','w','Color','none');
set(gca,'Color','k','XColor','w','YColor','w');

subplot(2,1,2);
plot(t_fmri, BF_tc, 'Color',[0.6 1 0.4], 'LineWidth',1.5); hold on;
plot(t_fmri, ach_ds, 'Color',[1 0.4 0.8], 'LineWidth',1.2);
title(sprintf('BF-like ROI vs ACh  (r=%.3f)', r_BF_ach), 'Color','w');
xlabel('Time (s)','Color','w'); ylabel('Z-score','Color','w');
legend('BF-like fMRI','ACh','TextColor','w','Color','none');
set(gca,'Color','k','XColor','w','YColor','w');

%% 5.5b Scrub outlier timepoints before ROI-NT correlations

% Identify outliers in either ROI timecourse
outlier_mask = abs(BF_tc) > 3.5 | abs(LC_tc) > 3.5;
clean_idx = ~outlier_mask;

fprintf('Scrubbing %d outlier timepoints: ', sum(outlier_mask));
disp(find(outlier_mask)');

% Recompute correlations on clean timepoints only
r_LC_ne_clean  = corr(LC_tc(clean_idx), ne_ds(clean_idx));
r_LC_ach_clean = corr(LC_tc(clean_idx), ach_ds(clean_idx));
r_BF_ne_clean  = corr(BF_tc(clean_idx), ne_ds(clean_idx));
r_BF_ach_clean = corr(BF_tc(clean_idx), ach_ds(clean_idx));

fprintf('\n--- ROI-NT coupling (scrubbed) ---\n');
fprintf('LC-like ROI ~ NE:  r = %.3f\n', r_LC_ne_clean);
fprintf('LC-like ROI ~ ACh: r = %.3f\n', r_LC_ach_clean);
fprintf('BF-like ROI ~ NE:  r = %.3f\n', r_BF_ne_clean);
fprintf('BF-like ROI ~ ACh: r = %.3f\n', r_BF_ach_clean);

% Replot with scrubbed timecourses
t_clean = t_fmri(clean_idx);
LC_clean = LC_tc(clean_idx);
BF_clean = BF_tc(clean_idx);
ne_clean  = ne_ds(clean_idx);
ach_clean = ach_ds(clean_idx);

figure('Color','k','Name','ROI Timecourses (scrubbed)');

subplot(2,1,1);
plot(t_clean, LC_clean, 'Color',[0.4 0.8 1], 'LineWidth',1.5); hold on;
plot(t_clean, ne_clean, 'Color',[1 0.6 0.2], 'LineWidth',1.2);
% Mark scrubbed timepoints
scrubbed_t = t_fmri(outlier_mask);
for i = 1:length(scrubbed_t)
    xline(scrubbed_t(i), 'r--', 'Alpha', 0.5);
end
title(sprintf('LC-like ROI vs NE  (r=%.3f, scrubbed)', r_LC_ne_clean),'Color','w');
xlabel('Time (s)','Color','w'); ylabel('Z-score','Color','w');
legend('LC-like fMRI','NE','TextColor','w','Color','none');
set(gca,'Color','k','XColor','w','YColor','w');

subplot(2,1,2);
plot(t_clean, BF_clean, 'Color',[0.6 1 0.4], 'LineWidth',1.5); hold on;
plot(t_clean, ach_clean, 'Color',[1 0.4 0.8], 'LineWidth',1.2);
for i = 1:length(scrubbed_t)
    xline(scrubbed_t(i), 'r--', 'Alpha', 0.5);
end
title(sprintf('BF-like ROI vs ACh  (r=%.3f, scrubbed)', r_BF_ach_clean),'Color','w');
xlabel('Time (s)','Color','w'); ylabel('Z-score','Color','w');
legend('BF-like fMRI','ACh','TextColor','w','Color','none');
set(gca,'Color','k','XColor','w','YColor','w');


%% Create brain mask from mean image

mean_brain = mean(V, 4);  % [102 x 197 x 75]

% Otsu's threshold to separate brain from background
mb_flat = mean_brain(:);
thresh = graythresh(mat2gray(mean_brain)) * max(mb_flat);

brain_mask_3D = mean_brain > thresh;

% Erode slightly to remove edge noise (manual 3D erosion)
brain_mask_3D = imerode(brain_mask_3D, strel('sphere', 2));

fprintf('Brain voxels: %d / %d total (%.1f%%)\n', ...
    sum(brain_mask_3D(:)), numel(brain_mask_3D), ...
    100*sum(brain_mask_3D(:))/numel(brain_mask_3D));

% Visualize mask on middle slices to check quality
mid_x = round(x/2); mid_y = round(y/2); mid_z = round(z/2);

figure('Color','k','Name','Brain mask check');
subplot(1,3,1);
imshow(squeeze(mb_norm(mid_x,:,:))' .* squeeze(double(brain_mask_3D(mid_x,:,:)))', []);
title(sprintf('Sagittal x=%d',mid_x),'Color','w');

subplot(1,3,2);
imshow(squeeze(mb_norm(:,mid_y,:))' .* squeeze(double(brain_mask_3D(:,mid_y,:)))', []);
title(sprintf('Coronal y=%d',mid_y),'Color','w');

subplot(1,3,3);
imshow(squeeze(mb_norm(:,:,mid_z))' .* squeeze(double(brain_mask_3D(:,:,mid_z)))', []);
title(sprintf('Axial z=%d',mid_z),'Color','w');
sgtitle('Brain mask check','Color','w');

%%
% Apply brain mask to ROI definitions
brain_mask_flat = brain_mask_3D(:);  % [nVoxels x 1]

LC_mask_masked = LC_mask & brain_mask_flat';
BF_mask_masked = BF_mask & brain_mask_flat';

fprintf('LC-like (brain only): %d voxels\n', sum(LC_mask_masked));
fprintf('BF-like (brain only): %d voxels\n', sum(BF_mask_masked));

%% 6. SPATIAL OVERLAY: LC-like and BF-like ROI anatomy

% mean_brain = mean(V, 4);  % [102 x 197 x 75]
% 
% % Reshape masks to 3D
% LC_3D = reshape(LC_mask, x, y, z);   % [102 x 197 x 75]
% BF_3D = reshape(BF_mask, x, y, z);
% 
% % Find peak slice for each ROI in all 3 orientations
% LC_z = squeeze(sum(sum(LC_3D, 1), 2));  % axial
% LC_y = squeeze(sum(sum(LC_3D, 1), 3));  % coronal
% LC_x = squeeze(sum(sum(LC_3D, 2), 3));  % sagittal
% 
% BF_z = squeeze(sum(sum(BF_3D, 1), 2));
% BF_y = squeeze(sum(sum(BF_3D, 1), 3));
% BF_x = squeeze(sum(sum(BF_3D, 2), 3));
% 
% [~, LC_peak_z] = max(LC_z); [~, LC_peak_y] = max(LC_y); [~, LC_peak_x] = max(LC_x);
% [~, BF_peak_z] = max(BF_z); [~, BF_peak_y] = max(BF_y); [~, BF_peak_x] = max(BF_x);
% 
% fprintf('LC-like ROI peak slices — x:%d  y:%d  z:%d\n', LC_peak_x, LC_peak_y, LC_peak_z);
% fprintf('BF-like ROI peak slices — x:%d  y:%d  z:%d\n', BF_peak_x, BF_peak_y, BF_peak_z);
% 
% % Normalize mean brain for display
%  mb_norm = mean_brain - min(mean_brain(:));
%  mb_norm = mb_norm / max(mb_norm(:));
% 
% % Color for overlays: LC=cyan, BF=lime
% LC_color = [0.0, 0.8, 1.0];
% BF_color = [0.4, 1.0, 0.2];
% 
% figure('Color','k','Name','ROI Anatomy','Position',[100 100 1400 800]);
% alpha_val = 0.55;
% 
% %--- Row 1: LC-like ROI (cyan) ---
% titles_lc = { sprintf('LC-like  |  Sagittal x=%d', LC_peak_x), ...
%               sprintf('LC-like  |  Coronal  y=%d', LC_peak_y), ...
%               sprintf('LC-like  |  Axial    z=%d', LC_peak_z) };
% 
% slices_lc = { squeeze(mb_norm(LC_peak_x,:,:))',  squeeze(LC_3D(LC_peak_x,:,:))'; ...
%               squeeze(mb_norm(:,LC_peak_y,:))',  squeeze(LC_3D(:,LC_peak_y,:))'; ...
%               squeeze(mb_norm(:,:,LC_peak_z))',  squeeze(LC_3D(:,:,LC_peak_z))' };
% 
% for col = 1:3
%     ax = subplot(2, 3, col);
%     brain_sl = slices_lc{col,1};
%     roi_sl   = slices_lc{col,2};
% 
%     % Build RGB image
%     rgb = repmat(brain_sl, [1 1 3]);
%     roi_rgb = zeros(size(brain_sl,1), size(brain_sl,2), 3);
%     for ch = 1:3
%         layer = zeros(size(brain_sl));
%         layer(roi_sl==1) = LC_color(ch);
%         roi_rgb(:,:,ch) = layer;
%     end
%     % Blend
%     alpha_map = double(roi_sl) * alpha_val;
%     for ch = 1:3
%         rgb(:,:,ch) = rgb(:,:,ch).*(1-alpha_map) + roi_rgb(:,:,ch).*alpha_map;
%     end
% 
%     imshow(rgb, []); axis image;
%     title(titles_lc{col}, 'Color','w', 'FontSize',11);
%     set(ax,'Color','k');
% end
% 
% %--- Row 2: BF-like ROI (lime) ---
% titles_bf = { sprintf('BF-like  |  Sagittal x=%d', BF_peak_x), ...
%               sprintf('BF-like  |  Coronal  y=%d', BF_peak_y), ...
%               sprintf('BF-like  |  Axial    z=%d', BF_peak_z) };
% 
% slices_bf = { squeeze(mb_norm(BF_peak_x,:,:))',  squeeze(BF_3D(BF_peak_x,:,:))'; ...
%               squeeze(mb_norm(:,BF_peak_y,:))',  squeeze(BF_3D(:,BF_peak_y,:))'; ...
%               squeeze(mb_norm(:,:,BF_peak_z))',  squeeze(BF_3D(:,:,BF_peak_z))' };
% 
% for col = 1:3
%     ax = subplot(2, 3, col+3);
%     brain_sl = slices_bf{col,1};
%     roi_sl   = slices_bf{col,2};
% 
%     rgb = repmat(brain_sl, [1 1 3]);
%     roi_rgb = zeros(size(brain_sl,1), size(brain_sl,2), 3);
%     for ch = 1:3
%         layer = zeros(size(brain_sl));
%         layer(roi_sl==1) = BF_color(ch);
%         roi_rgb(:,:,ch) = layer;
%     end
%     alpha_map = double(roi_sl) * alpha_val;
%     for ch = 1:3
%         rgb(:,:,ch) = rgb(:,:,ch).*(1-alpha_map) + roi_rgb(:,:,ch).*alpha_map;
%     end
% 
%     imshow(rgb, []); axis image;
%     title(titles_bf{col}, 'Color','w', 'FontSize',11);
%     set(ax,'Color','k');
% end
% 
% sgtitle('Anatomical location of NT-specific ROIs', 'Color','w', 'FontSize',13);

%% Redefine balanced ROIs within brain mask only

% Work only within brain voxels
brain_idx = find(brain_mask_flat);  % indices of brain voxels

rho_ach_brain = rho_ach(brain_idx);
rho_ne_brain  = rho_ne(brain_idx);

% Top 5% within brain voxels only
thresh_pct = 95;

ne_thresh_brain  = prctile(rho_ne_brain,  thresh_pct);
ach_thresh_brain = prctile(rho_ach_brain, 100 - thresh_pct); % most negative

% New masks (full voxel space, brain-constrained)
LC_mask_final = false(1, nV);
BF_mask_final = false(1, nV);

LC_mask_final(brain_idx) = rho_ne_brain  >= ne_thresh_brain;
BF_mask_final(brain_idx) = rho_ach_brain <= ach_thresh_brain;

fprintf('LC-like (brain, top 5%%): %d voxels\n', sum(LC_mask_final));
fprintf('BF-like (brain, top 5%%): %d voxels\n', sum(BF_mask_final));
fprintf('Overlap: %d voxels\n', sum(LC_mask_final & BF_mask_final));

% Recheck correlations with new masks
LC_tc_final = zscore(mean(fMRI_2D(:, LC_mask_final), 2));
BF_tc_final = zscore(mean(fMRI_2D(:, BF_mask_final), 2));

% Scrub outliers
outlier_mask2 = abs(BF_tc_final) > 3.5 | abs(LC_tc_final) > 3.5;
clean_idx2 = ~outlier_mask2;
fprintf('Scrubbing %d timepoints\n', sum(outlier_mask2));

r_LC_ne_final  = corr(LC_tc_final(clean_idx2), ne_ds(clean_idx2));
r_LC_ach_final = corr(LC_tc_final(clean_idx2), ach_ds(clean_idx2));
r_BF_ne_final  = corr(BF_tc_final(clean_idx2), ne_ds(clean_idx2));
r_BF_ach_final = corr(BF_tc_final(clean_idx2), ach_ds(clean_idx2));

fprintf('\n--- ROI-NT coupling (brain-masked, scrubbed) ---\n');
fprintf('LC-like ~ NE:  r = %.3f\n', r_LC_ne_final);
fprintf('LC-like ~ ACh: r = %.3f\n', r_LC_ach_final);
fprintf('BF-like ~ NE:  r = %.3f\n', r_BF_ne_final);
fprintf('BF-like ~ ACh: r = %.3f\n', r_BF_ach_final);

%% Spatial overlay with brain-masked ROIs
LC_3D_final = reshape(LC_mask_final, x, y, z);
BF_3D_final = reshape(BF_mask_final, x, y, z);

% Peak slices
[~, LC_peak_z] = max(squeeze(sum(sum(LC_3D_final,1),2)));
[~, LC_peak_y] = max(squeeze(sum(sum(LC_3D_final,1),3)));
[~, LC_peak_x] = max(squeeze(sum(sum(LC_3D_final,2),3)));

[~, BF_peak_z] = max(squeeze(sum(sum(BF_3D_final,1),2)));
[~, BF_peak_y] = max(squeeze(sum(sum(BF_3D_final,1),3)));
[~, BF_peak_x] = max(squeeze(sum(sum(BF_3D_final,2),3)));

fprintf('\nLC-like peak slices — x:%d  y:%d  z:%d\n', LC_peak_x, LC_peak_y, LC_peak_z);
fprintf('BF-like peak slices — x:%d  y:%d  z:%d\n', BF_peak_x, BF_peak_y, BF_peak_z);

LC_color = [0.0, 0.8, 1.0];  % cyan
BF_color = [0.4, 1.0, 0.2];  % lime

figure('Color','k','Name','ROI Anatomy (brain-masked)','Position',[100 100 1400 800]);
alpha_val = 0.6;

% Helper: build RGB overlay slice
make_rgb = @(brain_sl, roi_sl, col) ...
    cat(3, brain_sl.*(1-roi_sl*alpha_val) + col(1)*roi_sl*alpha_val, ...
           brain_sl.*(1-roi_sl*alpha_val) + col(2)*roi_sl*alpha_val, ...
           brain_sl.*(1-roi_sl*alpha_val) + col(3)*roi_sl*alpha_val);

% Row 1: LC-like
view_data = { ...
    squeeze(mb_norm(LC_peak_x,:,:))', squeeze(double(LC_3D_final(LC_peak_x,:,:)))', ...
    sprintf('LC-like | Sagittal x=%d', LC_peak_x);
    squeeze(mb_norm(:,LC_peak_y,:))', squeeze(double(LC_3D_final(:,LC_peak_y,:)))', ...
    sprintf('LC-like | Coronal  y=%d', LC_peak_y);
    squeeze(mb_norm(:,:,LC_peak_z))', squeeze(double(LC_3D_final(:,:,LC_peak_z)))', ...
    sprintf('LC-like | Axial    z=%d', LC_peak_z);
    squeeze(mb_norm(BF_peak_x,:,:))', squeeze(double(BF_3D_final(BF_peak_x,:,:)))', ...
    sprintf('BF-like | Sagittal x=%d', BF_peak_x);
    squeeze(mb_norm(:,BF_peak_y,:))', squeeze(double(BF_3D_final(:,BF_peak_y,:)))', ...
    sprintf('BF-like | Coronal  y=%d', BF_peak_y);
    squeeze(mb_norm(:,:,BF_peak_z))', squeeze(double(BF_3D_final(:,:,BF_peak_z)))', ...
    sprintf('BF-like | Axial    z=%d', BF_peak_z)};

for i = 1:6
    subplot(2,3,i);
    brain_sl = view_data{i,1};
    roi_sl   = view_data{i,2};
    col      = LC_color * (i<=3) + BF_color * (i>3);
    rgb      = make_rgb(brain_sl, roi_sl, col);
    imshow(rgb, []); axis image;
    title(view_data{i,3}, 'Color','w', 'FontSize',10);
end

sgtitle('NT-specific ROIs (brain-masked, top 5%)', 'Color','w', 'FontSize',13);


%% Distribution of ROI voxels across z-slices
figure('Color','k');
plot(1:z, LC_z_dist/max(LC_z_dist), 'c', 'LineWidth',1.5); hold on;
plot(1:z, BF_z_dist/max(BF_z_dist), 'm', 'LineWidth',1.5);
xlabel('Z slice','Color','w'); ylabel('Normalised voxel count','Color','w');
legend('LC-like','BF-like','TextColor','w','Color','none');
title('ROI distribution across z-slices','Color','w');
set(gca,'Color','k','XColor','w','YColor','w');


%% 
%% Fix slice selection — manually use brain extent thirds

z_lo  = 45;   % lower third of brain (LC-dominant per z-distribution)
z_mid = 51;   % middle
z_hi  = 60;   % upper third (BF-dominant per z-distribution)

z_slices = [z_lo, z_mid, z_hi];
fprintf('Display slices: %d, %d, %d\n', z_slices);

% % Find crossover properly — only within brain z extent
% z_brain_range = 15:67;
% LC_in_range = LC_z_dist_final(z_brain_range);
% BF_in_range = BF_z_dist_final(z_brain_range);
% diff_in_range = LC_in_range - BF_in_range;
% 
% % Find where LC-BF changes sign (LC dominant → BF dominant)
% sign_changes = find(diff(sign(diff_in_range)));
% if ~isempty(sign_changes)
%     crossover_z_fixed = z_brain_range(sign_changes(1));
%     fprintf('LC/BF crossover (within brain): z=%d\n', crossover_z_fixed);
%     z_slices(2) = crossover_z_fixed;  % replace mid with true crossover
% end
% 
% fprintf('Final display slices: %d, %d, %d\n', z_slices);

%% Plot
figure('Color','k','Name','Combined ROI overlay','Position',[100 100 1500 550]);

[bx, by, ~] = ind2sub([x,y,z], find(brain_mask_3D));
x_lo = max(1, min(bx)-3); x_hi = min(x, max(bx)+3);
y_lo = max(1, min(by)-3); y_hi = min(y, max(by)+3);

labels = {'Inferior (LC-dominant)', 'Mid (transition)', 'Superior (BF-dominant)'};

for i = 1:3
    zs = z_slices(i);
    
    brain_sl = squeeze(mb_norm(:,:,zs))';
    LC_sl    = squeeze(double(LC_3D_final(:,:,zs)))';
    BF_sl    = squeeze(double(BF_3D_final(:,:,zs)))';
    
    % Crop to brain bounding box
    brain_sl = brain_sl(y_lo:y_hi, x_lo:x_hi);
    LC_sl    = LC_sl(y_lo:y_hi,    x_lo:x_hi);
    BF_sl    = BF_sl(y_lo:y_hi,    x_lo:x_hi);
    
    % Build RGB — BF first, LC on top, overlap white
    R = brain_sl; G = brain_sl; B = brain_sl;
    
    R(BF_sl==1) = 0.2;  G(BF_sl==1) = 0.9;  B(BF_sl==1) = 0.1;
    R(LC_sl==1) = 0.0;  G(LC_sl==1) = 0.8;  B(LC_sl==1) = 1.0;
    
    ov = LC_sl & BF_sl;
    R(ov==1) = 1.0; G(ov==1) = 1.0; B(ov==1) = 1.0;
    
    rgb = cat(3, R, G, B);
    
    subplot(1,3,i);
    imshow(rgb, []); axis image off;
    
    lc_n = sum(LC_sl(:)); bf_n = sum(BF_sl(:));
    title({labels{i}, sprintf('z=%d  |  LC:%d  BF:%d', zs, lc_n, bf_n)}, ...
          'Color','w','FontSize',10);
end

subplot(1,3,1); hold on;
text(3, 8,  '■ LC-like / NE',  'Color',[0.0 0.8 1.0],'FontSize',11,'FontWeight','bold');
text(3, 18, '■ BF-like / ACh', 'Color',[0.2 0.9 0.1],'FontSize',11,'FontWeight','bold');
text(3, 28, '■ Overlap',       'Color',[1.0 1.0 1.0],'FontSize',11,'FontWeight','bold');

sgtitle('NT-specific ROIs: inferior-to-superior gradient', ...
        'Color','w','FontSize',13,'FontWeight','bold');

%% Clean gradient plot — no ratio, just smooth distributions

% Smooth the distributions for cleaner visualization
smooth_win = 5;
LC_z_smooth = movmean(LC_z, smooth_win);
BF_z_smooth = movmean(BF_z, smooth_win);

% Normalize
LC_z_norm = LC_z_smooth / max(LC_z_smooth);
BF_z_norm = BF_z_smooth / max(BF_z_smooth);

% Find crossover within brain (where BF overtakes LC)
diff_norm = LC_z_norm - BF_z_norm;
sign_ch = find(diff(sign(diff_norm)) < 0);  % LC→BF crossover only
if ~isempty(sign_ch)
    cross_z = z_range(sign_ch(end));  % last crossover
    fprintf('LC→BF crossover at z=%d\n', cross_z);
end

figure('Color','k','Name','NT gradient clean','Position',[100 100 900 500]);

% Shade regions
patch([z_range(1) cross_z cross_z z_range(1)], [0 0 1 1], ...
      'c', 'FaceAlpha', 0.08, 'EdgeColor','none'); hold on;
patch([cross_z z_range(end) z_range(end) cross_z], [0 0 1 1], ...
      [0.2 0.9 0.1], 'FaceAlpha', 0.08, 'EdgeColor','none');

% Plot smoothed distributions
plot(z_range, LC_z_norm, 'c',  'LineWidth', 2.5);
plot(z_range, BF_z_norm, 'Color',[0.2 0.9 0.1], 'LineWidth', 2.5);

% Crossover line
xline(cross_z, 'w--', sprintf('z=%d', cross_z), ...
      'LabelColor','w', 'LineWidth', 1.5, 'LabelVerticalAlignment','bottom');

% Update slice markers to reflect true crossover
xline(45, 'w:', 'z=45', 'LabelColor','w', 'LineWidth',1, 'Alpha',0.4);
xline(51, 'w:', 'z=51 (crossover)', 'LabelColor','w', 'LineWidth',1.5);
xline(60, 'w:', 'z=60', 'LabelColor','w', 'LineWidth',1, 'Alpha',0.4);

% Region labels
text(z_range(1)+1, 0.92, 'LC-dominant', 'Color','c', ...
     'FontSize',10, 'FontWeight','bold');
text(cross_z+1,    0.92, 'BF-dominant', 'Color',[0.2 0.9 0.1], ...
     'FontSize',10, 'FontWeight','bold');

xlabel('Z slice (inferior → superior)', 'Color','w', 'FontSize',12);
ylabel('Normalised voxel density',      'Color','w', 'FontSize',12);
title({'Spatial gradient of NT-specific ROIs', ...
       'LC-like broadly distributed; BF-like concentrated superiorly'}, ...
      'Color','w', 'FontSize',11);
legend('','','LC-like (NE)','BF-like (ACh)', ...
       'TextColor','w','Color','none','Location','northwest');
set(gca,'Color','k','XColor','w','YColor','w','FontSize',11);
ylim([0 1.05]);

%%
figure('Color','k','Name','LC:BF ratio across z','Position',[100 100 800 500]);

% Plot normalized distributions
yyaxis left
plot(z_range, LC_z/max(LC_z), 'c', 'LineWidth', 2); hold on;
plot(z_range, BF_z/max(BF_z), 'Color',[0.2 0.9 0.1], 'LineWidth', 2);
ylabel('Normalised voxel density','Color','w');
set(gca,'YColor','w');

yyaxis right
plot(z_range, ratio, 'w--', 'LineWidth', 1);
yline(1, 'r:', 'LineWidth', 1.5);   % LC=BF line
ylabel('LC:BF ratio','Color','w');
set(gca,'YColor','w');

% Shade LC-dominant region
lc_dom = z_range(ratio >= 1);
if ~isempty(lc_dom)
    patch([lc_dom(1) lc_dom(end) lc_dom(end) lc_dom(1)], ...
          [0 0 1 1]*max(ratio)*1.1, 'c', ...
          'FaceAlpha',0.08, 'EdgeColor','none');
end

xlabel('Z slice (inferior → superior)','Color','w');
title({'Spatial gradient of NT-specific ROIs', ...
       'LC-like broadly distributed; BF-like concentrated superiorly'}, ...
      'Color','w','FontSize',11);
legend('LC-like (NE)','BF-like (ACh)','LC:BF ratio','LC=BF', ...
       'TextColor','w','Color','none','Location','northwest');
set(gca,'Color','k','XColor','w','FontSize',11);

% Add z slice markers
xline(45,'w:','z=45','LabelColor','w','Alpha',0.5);
xline(51,'w:','z=51','LabelColor','w','Alpha',0.5);
xline(60,'w:','z=60','LabelColor','w','Alpha',0.5);

%%
%% 7. ON/OFF STATE ANALYSIS: Which voxels are most coordinated?

% 7.1 Define ON/OFF states via median split of pupil z-score
pupil_median = median(pupil_fmri);
ON_idx  = pupil_fmri >= pupil_median;   % large pupil = aroused = ON
OFF_idx = pupil_fmri <  pupil_median;   % small pupil = quiet = OFF

fprintf('ON  timepoints: %d\n', sum(ON_idx));
fprintf('OFF timepoints: %d\n', sum(OFF_idx));

% 7.2 Voxelwise two-sample t-test: ON vs OFF BOLD
% Only run within brain mask to save memory
nV_brain = sum(brain_mask_flat);
brain_vox_idx = find(brain_mask_flat);

t_stat  = zeros(1, nV);   % full voxel space, fill brain voxels only
p_val   = ones(1, nV);

fprintf('Running voxelwise t-tests (%d brain voxels)...\n', nV_brain);

for vi = 1:nV_brain
    v = brain_vox_idx(vi);
    on_bold  = fMRI_2D(ON_idx,  v);
    off_bold = fMRI_2D(OFF_idx, v);
    
    % Manual two-sample t-test (equal n approximately)
    n1 = sum(ON_idx);  n2 = sum(OFF_idx);
    m1 = mean(on_bold); m2 = mean(off_bold);
    s1 = std(on_bold);  s2 = std(off_bold);
    
    sp = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2));
    t_stat(v) = (m1 - m2) / (sp * sqrt(1/n1 + 1/n2));
    
    % Two-tailed p-value from t-distribution
    df = n1 + n2 - 2;
    p_val(v) = 2 * (1 - tcdf(abs(t_stat(v)), df));
end

fprintf('Done.\n');

% 7.3 Threshold and find peak voxels
% FDR correction (Benjamini-Hochberg) within brain
p_brain = p_val(brain_vox_idx);
[p_sorted, sort_idx] = sort(p_brain);
m = length(p_brain);
fdr_thresh = 0.05;
bh_thresh  = (1:m)' / m * fdr_thresh;
sig_fdr    = p_sorted <= bh_thresh';
if any(sig_fdr)
    p_crit = p_sorted(find(sig_fdr, 1, 'last'));
else
    p_crit = 0.001;  % fallback uncorrected
    fprintf('No FDR-significant voxels, using p<0.001 uncorrected\n');
end
fprintf('FDR p-critical: %.6f\n', p_crit);

% Significant voxels
sig_mask = p_val <= p_crit & brain_mask_flat';
ON_mask  = sig_mask & t_stat > 0;   % ON > OFF
OFF_mask = sig_mask & t_stat < 0;   % OFF > ON

fprintf('Significant ON  voxels: %d\n', sum(ON_mask));
fprintf('Significant OFF voxels: %d\n', sum(OFF_mask));

% 7.4 Print top 10 ON and OFF voxels with coordinates
t_ON  = t_stat .* ON_mask;
t_OFF = t_stat .* OFF_mask;

[~, ON_sorted]  = sort(t_ON,  'descend');
[~, OFF_sorted] = sort(t_OFF, 'ascend');   % most negative first

fprintf('\n--- Top 10 ON-state voxels (BOLD higher during large pupil) ---\n');
fprintf('%-8s %-6s %-6s %-6s %-10s %-10s\n','Rank','X','Y','Z','t-stat','p-val');
for k = 1:10
    v = ON_sorted(k);
    if ~ON_mask(v), break; end
    [vx,vy,vz] = ind2sub([x,y,z], v);
    fprintf('%-8d %-6d %-6d %-6d %-10.3f %-10.6f\n', ...
            k, vx, vy, vz, t_stat(v), p_val(v));
end

fprintf('\n--- Top 10 OFF-state voxels (BOLD higher during small pupil) ---\n');
fprintf('%-8s %-6s %-6s %-6s %-10s %-10s\n','Rank','X','Y','Z','t-stat','p-val');
for k = 1:10
    v = OFF_sorted(k);
    if ~OFF_mask(v), break; end
    [vx,vy,vz] = ind2sub([x,y,z], v);
    fprintf('%-8d %-6d %-6d %-6d %-10.3f %-10.6f\n', ...
            k, vx, vy, vz, t_stat(v), p_val(v));
end

% 7.5 Visualize t-map on brain
t_3D = reshape(t_stat, x, y, z);

% Show on three axial slices
z_display = [25, 51, 60];
figure('Color','k','Name','ON vs OFF t-map','Position',[100 100 1400 500]);

for i = 1:3
    zs = z_display(i);
    brain_sl = squeeze(mb_norm(:,:,zs))';
    t_sl     = squeeze(t_3D(:,:,zs))';
    sig_3D_temp = reshape(sig_mask, x, y, z);   % separate line
    sig_sl   = squeeze(sig_3D_temp(:,:,zs))';
    
    % Crop
    brain_sl = brain_sl(y_lo:y_hi, x_lo:x_hi);
    t_sl     = t_sl(y_lo:y_hi,     x_lo:x_hi);
    sig_sl   = sig_sl(y_lo:y_hi,   x_lo:x_hi);
    
    % Build RGB: red=ON, blue=OFF, only significant
    R = brain_sl; G = brain_sl; B = brain_sl;
    
    t_max = max(abs(t_stat(sig_mask)));  % for scaling
    
    ON_sl  = sig_sl & t_sl > 0;
    OFF_sl = sig_sl & t_sl < 0;
    
    % Scale color intensity by t-statistic magnitude
    t_norm = abs(t_sl) / t_max;
    
    R(ON_sl)  = 0.4 + 0.6*t_norm(ON_sl);
    G(ON_sl)  = 0;
    B(ON_sl)  = 0;
    
    R(OFF_sl) = 0;
    G(OFF_sl) = 0;
    B(OFF_sl) = 0.4 + 0.6*t_norm(OFF_sl);
    
    rgb = cat(3,R,G,B);
    
    subplot(1,3,i);
    imshow(rgb,[]); axis image off;
    title(sprintf('z=%d', zs),'Color','w','FontSize',11);
end

subplot(1,3,1); hold on;
text(3, 8,  '■ ON-state  (BOLD↑ pupil↑)', 'Color',[1 0.3 0.3],'FontSize',9,'FontWeight','bold');
text(3, 18, '■ OFF-state (BOLD↑ pupil↓)', 'Color',[0.3 0.3 1],'FontSize',9,'FontWeight','bold');
sgtitle('Voxelwise ON vs OFF state BOLD (FDR corrected)','Color','w','FontSize',12);

%% 7.6 Save NIfTI using SPM (replacing make_nii)

% Build a minimal SPM volume header
V_hdr        = struct();
V_hdr.dim    = [x, y, z];
V_hdr.dt     = [spm_type('float32'), spm_platform('bigend')];
V_hdr.mat    = eye(4);   % identity — native voxel space, no MNI
V_hdr.pinfo  = [1; 0; 0];
V_hdr.descrip = '';

% Reshape masks to 3D before saving
ON_3D  = reshape(ON_mask,  x, y, z);
OFF_3D = reshape(OFF_mask, x, y, z);
t_3D   = reshape(t_stat,   x, y, z);

% Save t-map
V_hdr.fname   = 'on_off_tmap.nii';
V_hdr.descrip = 'ON vs OFF t-map (pupil median split)';
spm_write_vol(V_hdr, t_3D);
fprintf('Saved: on_off_tmap.nii\n');

% Save ON-state mask
V_hdr.fname   = 'on_state_mask.nii';
V_hdr.descrip = 'ON-state voxels (BOLD higher, large pupil)';
spm_write_vol(V_hdr, double(ON_3D));
fprintf('Saved: on_state_mask.nii\n');

% Save OFF-state mask
V_hdr.fname   = 'off_state_mask.nii';
V_hdr.descrip = 'OFF-state voxels (BOLD higher, small pupil)';
spm_write_vol(V_hdr, double(OFF_3D));
fprintf('Saved: off_state_mask.nii\n');


%% Interactive slice viewer — scroll through z slices
figure('Color','k');
for zs = 15:67
    brain_sl = squeeze(mb_norm(:,:,zs))';
    t_sl     = squeeze(t_3D(:,:,zs))';
    sig_sl   = squeeze(sig_3D_temp(:,:,zs))';
    
    brain_sl = brain_sl(y_lo:y_hi, x_lo:x_hi);
    t_sl     = t_sl(y_lo:y_hi,     x_lo:x_hi);
    sig_sl   = sig_sl(y_lo:y_hi,   x_lo:x_hi);
    
    R = brain_sl; G = brain_sl; B = brain_sl;
    
    ON_sl  = sig_sl & t_sl > 0;
    OFF_sl = sig_sl & t_sl < 0;
    t_max  = max(abs(t_stat(sig_mask)));
    t_norm = abs(t_sl) / t_max;
    
    R(ON_sl)  = 0.4 + 0.6*t_norm(ON_sl);
    G(ON_sl)  = 0; B(ON_sl)  = 0;
    R(OFF_sl) = 0; G(OFF_sl) = 0;
    B(OFF_sl) = 0.4 + 0.6*t_norm(OFF_sl);
    
    imshow(cat(3,R,G,B),[]); axis image off;
    title(sprintf('z=%d  |  ON(red):%d  OFF(blue):%d', ...
          zs, sum(ON_sl(:)), sum(OFF_sl(:))), 'Color','w');
    drawnow; pause(0.5);
end


%% 8. VOXELWISE LAG MAP: Does BOLD-pupil lag vary by brain region?

% Pre-filter signals before voxelwise lag analysis
fs = 1/TR;

% Bandpass: keep 0.01–0.1 Hz arousal band
[b_bp, a_bp] = butter(3, [0.01 0.1]/(fs/2), 'bandpass');

pupil_filt = filtfilt(b_bp, a_bp, pupil_fmri);

% Filter each brain voxel timecourse
fMRI_filt = zeros(size(fMRI_2D));
fprintf('Bandpass filtering %d brain voxels...\n', nV_brain);
for vi = 1:nV_brain
    v = brain_idx(vi);
    fMRI_filt(:,v) = filtfilt(b_bp, a_bp, fMRI_2D(:,v));
end
fprintf('Done.\n');

max_lag = 8;   % ±12 TRs = ±22.5 s search window
nV_brain = sum(brain_mask_flat);
brain_idx = find(brain_mask_flat);

% Pre-allocate
peak_lag_map  = NaN(1, nV);   % peak lag in TRs
peak_corr_map = zeros(1, nV); % correlation at peak lag

fprintf('Computing voxelwise lag maps (%d brain voxels)...\n', nV_brain);

for vi = 1:nV_brain
    v = brain_idx(vi);
    ts = fMRI_2D(:, v);
    
    % Skip flat/dead voxels
    if std(ts) < 1e-6, continue; end
    
    [xc, lags_v] = xcorr(ts, pupil_fmri, max_lag, 'coeff');
    
    % Find peak — separately for positive and negative peaks
    [max_pos, idx_pos] = max(xc);
    [max_neg, idx_neg] = min(xc);
    
    % Take whichever peak is larger in magnitude
    if abs(max_pos) >= abs(max_neg)
        peak_lag_map(v)  = lags_v(idx_pos);
        peak_corr_map(v) = max_pos;
    else
        peak_lag_map(v)  = lags_v(idx_neg);
        peak_corr_map(v) = max_neg;
    end
end

fprintf('Done.\n');

% Reshape to 3D
lag_3D  = reshape(peak_lag_map,  x, y, z);
corr_3D = reshape(peak_corr_map, x, y, z);

% Convert lag from TRs to seconds
lag_3D_s = lag_3D * TR;

%% Summary statistics by region
% Compare lag distributions: LC-like vs BF-like vs ON vs OFF voxels
lag_LC  = peak_lag_map(LC_mask_final) * TR;
lag_BF  = peak_lag_map(BF_mask_final) * TR;
lag_ON  = peak_lag_map(ON_mask)  * TR;
lag_OFF = peak_lag_map(OFF_mask) * TR;
lag_all_brain = peak_lag_map(brain_idx) * TR;

fprintf('\n--- Lag statistics by region (seconds) ---\n');
fprintf('All brain:    mean=%.1f s, std=%.1f s\n', mean(lag_all_brain,'omitnan'), std(lag_all_brain,'omitnan'));
fprintf('LC-like ROI:  mean=%.1f s, std=%.1f s\n', mean(lag_LC,'omitnan'),  std(lag_LC,'omitnan'));
fprintf('BF-like ROI:  mean=%.1f s, std=%.1f s\n', mean(lag_BF,'omitnan'),  std(lag_BF,'omitnan'));
fprintf('ON voxels:    mean=%.1f s, std=%.1f s\n', mean(lag_ON,'omitnan'),  std(lag_ON,'omitnan'));
fprintf('OFF voxels:   mean=%.1f s, std=%.1f s\n', mean(lag_OFF,'omitnan'), std(lag_OFF,'omitnan'));

% KS tests between groups
[~, p_LC_BF]  = kstest2(lag_LC(~isnan(lag_LC)),   lag_BF(~isnan(lag_BF)));
[~, p_ON_OFF] = kstest2(lag_ON(~isnan(lag_ON)),   lag_OFF(~isnan(lag_OFF)));
fprintf('\nKS test LC vs BF lag distributions:  p = %.4f\n', p_LC_BF);
fprintf('KS test ON vs OFF lag distributions: p = %.4f\n', p_ON_OFF);

%% Figure 1: Lag distribution histograms
figure('Color','k','Name','Lag distributions','Position',[100 100 1000 500]);

subplot(1,2,1);
edges = (-max_lag:max_lag) * TR;
histogram(lag_LC(~isnan(lag_LC)),  edges, 'FaceColor','c',          'FaceAlpha',0.6,'EdgeColor','none'); hold on;
histogram(lag_BF(~isnan(lag_BF)),  edges, 'FaceColor',[0.2 0.9 0.1],'FaceAlpha',0.6,'EdgeColor','none');
xline(0,'w--','LineWidth',1.5);
xline(6,'r:','6s HRF','LabelColor','r','LineWidth',1);
xlabel('Peak lag (s)','Color','w'); ylabel('Voxel count','Color','w');
title(sprintf('LC vs BF lag (KS p=%.4f)',p_LC_BF),'Color','w');
legend('LC-like','BF-like','TextColor','w','Color','none');
set(gca,'Color','k','XColor','w','YColor','w');

subplot(1,2,2);
histogram(lag_ON(~isnan(lag_ON)),  edges, 'FaceColor',[1 0.3 0.3], 'FaceAlpha',0.6,'EdgeColor','none'); hold on;
histogram(lag_OFF(~isnan(lag_OFF)), edges, 'FaceColor',[0.3 0.3 1], 'FaceAlpha',0.6,'EdgeColor','none');
xline(0,'w--','LineWidth',1.5);
xline(6,'r:','6s HRF','LabelColor','r','LineWidth',1);
xlabel('Peak lag (s)','Color','w'); ylabel('Voxel count','Color','w');
title(sprintf('ON vs OFF lag (KS p=%.4f)',p_ON_OFF),'Color','w');
legend('ON voxels','OFF voxels','TextColor','w','Color','none');
set(gca,'Color','k','XColor','w','YColor','w');
sgtitle('BOLD-pupil peak lag distributions by ROI','Color','w','FontSize',12);

%% Figure 2: Lag map overlaid on brain slices
% Colormap: blue=negative lag (region leads pupil), red=positive lag (region follows)
z_display = [25, 51, 60];

figure('Color','k','Name','Voxelwise lag map','Position',[100 100 1400 500]);

for i = 1:3
    zs = z_display(i);
    
    brain_sl = squeeze(mb_norm(:,:,zs))';
    lag_sl   = squeeze(lag_3D_s(:,:,zs))';
    mask_sl  = squeeze(brain_mask_3D(:,:,zs))';
    
    % Crop
    brain_sl = brain_sl(y_lo:y_hi, x_lo:x_hi);
    lag_sl   = lag_sl(y_lo:y_hi,   x_lo:x_hi);
    mask_sl  = mask_sl(y_lo:y_hi,  x_lo:x_hi);
    
    % Normalize lag to [0 1] for colormap: 0=most negative, 0.5=zero, 1=most positive
    lag_norm = (lag_sl - (-max_lag*TR)) / (2*max_lag*TR);
    lag_norm = max(0, min(1, lag_norm));
    
    % Build RGB: blue→white→red colormap
    R = zeros(size(lag_sl)); G = zeros(size(lag_sl)); B = zeros(size(lag_sl));
    
    % Negative lags → blue to white
    neg = lag_norm < 0.5;
    t_neg = lag_norm(neg) * 2;   % 0→1 as lag goes from -max to 0
    R(neg) = t_neg; G(neg) = t_neg; B(neg) = 1;
    
    % Positive lags → white to red
    pos = lag_norm >= 0.5;
    t_pos = (lag_norm(pos) - 0.5) * 2;  % 0→1 as lag goes from 0 to +max
    R(pos) = 1; G(pos) = 1-t_pos; B(pos) = 1-t_pos;
    
    % Gray background for non-brain
    R(~mask_sl) = brain_sl(~mask_sl) * 0.6;
    G(~mask_sl) = brain_sl(~mask_sl) * 0.6;
    B(~mask_sl) = brain_sl(~mask_sl) * 0.6;
    
    rgb = cat(3,R,G,B);
    
    subplot(1,3,i);
    imshow(rgb,[]); axis image off;
    title(sprintf('z=%d', zs),'Color','w','FontSize',11);
end

% Colorbar legend
subplot(1,3,1); hold on;
text(3, 8,  sprintf('Blue = early lag (~-%ds)',max_lag), 'Color',[0.4 0.4 1],'FontSize',9,'FontWeight','bold');
text(3, 18, 'White = ~0 s lag',                          'Color','w','FontSize',9,'FontWeight','bold');
text(3, 28, sprintf('Red = late lag (~+%ds)',max_lag),   'Color',[1 0.4 0.4],'FontSize',9,'FontWeight','bold');
text(3, 38, 'Red dot = canonical HRF (6s)',              'Color','r','FontSize',9);

sgtitle('Voxelwise BOLD-pupil peak lag map','Color','w','FontSize',13);

%% Figure 3: Lag vs z-slice — does lag vary with brain depth?
lag_by_z = zeros(1, z);
for zs = 1:z
    mask_z = squeeze(brain_mask_3D(:,:,zs));
    lag_z  = squeeze(lag_3D_s(:,:,zs));
    vals   = lag_z(mask_z);
    if ~isempty(vals)
        lag_by_z(zs) = mean(vals, 'omitnan');
    end
end

figure('Color','k','Name','Lag vs z-slice','Position',[100 100 800 400]);
z_range_plot = 15:67;
lag_smooth = movmean(lag_by_z(z_range_plot), 5);

plot(z_range_plot, lag_smooth, 'w', 'LineWidth', 2); hold on;
yline(6,  'r--', 'Canonical HRF (6s)', 'LabelColor','r','LineWidth',1.5);
yline(0,  'w:',  'LineWidth', 1);
xline(51, 'c--', 'LC/BF crossover z=51', 'LabelColor','c','LineWidth',1);

% Shade by NT territory
patch([15 51 51 15], [-max_lag*TR -max_lag*TR max_lag*TR max_lag*TR], ...
      'c', 'FaceAlpha',0.06,'EdgeColor','none');
patch([51 67 67 51], [-max_lag*TR -max_lag*TR max_lag*TR max_lag*TR], ...
      [0.2 0.9 0.1], 'FaceAlpha',0.06,'EdgeColor','none');

text(17, max_lag*TR*0.85, 'LC territory','Color','c','FontSize',10,'FontWeight','bold');
text(53, max_lag*TR*0.85, 'BF territory','Color',[0.2 0.9 0.1],'FontSize',10,'FontWeight','bold');

xlabel('Z slice (inferior → superior)','Color','w','FontSize',11);
ylabel('Mean BOLD-pupil peak lag (s)','Color','w','FontSize',11);
title('Does BOLD-pupil lag vary across brain depth?','Color','w','FontSize',12);
set(gca,'Color','k','XColor','w','YColor','w','FontSize',11);
ylim([-max_lag*TR, max_lag*TR]);