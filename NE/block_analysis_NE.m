clear
load ("NE_data.mat");

%% BINNING
% Bin pupil_z data to 1s binsx
bins = 1:trace_end_s;
pupil_binned = zeros(trace_end_s,1);

for bin = bins
    pupil_binned(bin) = mean(pupil_z((bin-1)*fs_pupil + 1:bin*fs_pupil));
end

% Compute threshold for pupil_z size
threshold = mean(pupil_z);

% Identify pupil_z data above the threshold
pupil_above_threshold = pupil_binned > threshold;

% Find time indices of NE block
block_start_s = find(diff(pupil_above_threshold) ~= 0)+1;
block_end_s = block_start_s-1;
block_start_idx = block_start_s * fs_pupil;
block_end_idx = block_end_s * fs_pupil;
block_start_idx = [1; block_start_idx];
block_end_idx = [block_end_idx; trace_end];
blocks_T = table(block_start_idx, block_end_idx,'VariableNames',{'block_start_idx', 'block_end_idx'});

% Add state duration
blocks_T.duration = (blocks_T.block_end_idx - blocks_T.block_start_idx)/fs_pupil;

% Assign state to block
blocks_T.pupil_state = zeros(height(blocks_T), 1);
first_block = pupil_above_threshold(1);

if first_block == 0
    block_on = 2:2:height(blocks_T);
else
    block_on = 1:2:height(blocks_T);
end

blocks_T.pupil_state(block_on) = 1;

% Compute threshold
threshold = mean(pupil_z);

% If blocks_T is a table, convert to numeric matrix
if istable(blocks_T)
    blocks = [blocks_T.block_start_idx, blocks_T.block_end_idx, blocks_T.pupil_state];
else
    blocks = blocks_T;
end

% Plot signal
figure(2);
p1 = plot(time,pupil_z,"Color",[0.5 0.5 0.5], "LineWidth",0.7);
hold on;
p2 = plot(time,NE_z,'Color', RGB(3,:), 'LineWidth',0.7);
box off;
yl = ylim;
xlim = [0 300];
p3 = yline(threshold,'r--');
legend([p1, p2, p3], {'Pupil size (Z-scored)', 'NE (Z-scored)', 'Pupil threshold'});

% Shade NE block where pupil_state == 1
maskRows = blocks(:,3) == 1;
starts = blocks(maskRows,1);
ends   = blocks(maskRows,2);

for k = 1:numel(starts)
    % guard against out-of-range indices
    s = max(1, round(starts(k)));
    e = min(numel(time), round(ends(k)));
    if e < s, continue; end

    xpatch = [time(s) time(e) time(e) time(s)];
    ypatch = [yl(1) yl(1) yl(2) yl(2)];
    h = patch(xpatch, ypatch, [0.5 0.5 0.5], 'FaceAlpha', 0.07, ...
              'EdgeColor', 'none');
    uistack(h,'bottom'); % put patches behind the trace
    h.HandleVisibility = 'off';
end

hold off;
xlabel('Time (s)');
ylabel('Z-score');
title('NE trace during identified brain states');
exportgraphics(gcf, fullfile('figures','figure2_normNE.jpg'), 'Resolution', 300);

% Gather data
blocks_T.pupil_data_z = cell(height(blocks_T),1);
blocks_T.nt_data_z = cell(height(blocks_T),1);


for block = 1:height(blocks_T)
    s = blocks_T.block_start_idx(block);
    e = blocks(block, 2);
    blocks_T.pupil_data_z{block} = pupil_z(s:e);
    blocks_T.nt_data_z{block} = NE_z(s:e);
end

%% ANALYSIS: Correlation
% Compute correlation
blocks_T.corr = cell(height(blocks_T),1);
for block = 1:height(blocks_T)
    corr = corrcoef(blocks_T.pupil_data_z{block},blocks_T.nt_data_z{block});
    corr = unique(corr(find(corr ~= 1)));
    blocks_T.corr{block} = corr;
end

off_corr = blocks_T.corr(blocks_T.pupil_state == 0 & blocks_T.duration > 1);
off_corr = cell2mat(off_corr(~cellfun('isempty', off_corr))); %remove invalid data points
on_corr = blocks_T.corr(blocks_T.pupil_state == 1 & blocks_T.duration > 1);
on_corr = cell2mat(on_corr(~cellfun('isempty', on_corr))); %remove invalid data points

% Fisher transform correlations
z_off = atanh(off_corr);
z_on  = atanh(on_corr);

% Descriptive statistics
off_mean_corr = mean(off_corr, 'omitnan');
on_mean_corr  = mean(on_corr, 'omitnan');

off_mean_z = mean(z_off, 'omitnan');
on_mean_z  = mean(z_on, 'omitnan');

% Permutation test
combined = [z_off(:); z_on(:)];
labels = [zeros(numel(z_off),1); ones(numel(z_on),1)];

n_perm = 10000;

diff_real = mean(z_off, 'omitnan') - mean(z_on, 'omitnan');
diff_perm = zeros(n_perm,1);

for i = 1:n_perm
    perm_labels = labels(randperm(numel(labels)));

    diff_perm(i) = mean(combined(perm_labels == 0), 'omitnan') - ...
                   mean(combined(perm_labels == 1), 'omitnan');
end

p_state_corr_perm = mean(abs(diff_perm) >= abs(diff_real));

% Print and save results

fprintf('off_mean_corr = %.4f, n = %d\n', off_mean_corr, numel(off_corr));
fprintf('on_mean_corr = %.4f, n = %d\n', on_mean_corr, numel(on_corr));
fprintf('Permutation test: diff = %.4f, p = %.4f\n', diff_real, p_state_corr_perm);
save("NE_results.mat", 'off_mean_corr','on_mean_corr', 'diff_real','p_state_corr_perm','-append');

figure(3)
group = [repmat("off",numel(off_corr),1); repmat("on",numel(on_corr),1)];
boxplot([off_corr; on_corr], group, 'Notch','on', 'Labels', {'OFF','ON'})
ylabel("Mean Pearson's Correlation")
title('NE Correlation during ON and OFF brain states')
box off;
hold off;
exportgraphics(gcf, fullfile('figures','figure3_corrNE.jpg'), 'Resolution', 300);

%% ANALYSIS (Peri-event)
blocks_T.peri_start_trace = cell(height(blocks_T),1);

onset = blocks_T.block_start_idx(1:end);
time_padding_s = 10;
pre_onset = onset - (time_padding_s*fs_pupil);
post_onset = onset + (time_padding_s*fs_pupil);

% Initialize peri-event traces
for block = 2:height(blocks_T)
    peri_start = pre_onset(block);
    peri_end = post_onset(block);
    blocks_T.peri_start_trace{block} = pupil_z(peri_start:peri_end);
end

% Filter valid blocks
off_peri = blocks_T.peri_start_trace(blocks_T.pupil_state == 0 & blocks_T.duration > time_padding_s);
off_peri = horzcat(off_peri{:})';
mean_off_peri = mean(off_peri, 1);

on_peri = blocks_T.peri_start_trace(blocks_T.pupil_state == 1 & blocks_T.duration > time_padding_s);
on_peri = horzcat(on_peri{:})';
mean_on_peri = mean(on_peri, 1);

% Time vector (match data length)
n = size(off_peri, 2);
time_peri = linspace(-time_padding_s, time_padding_s, n);

% Plot
figure(4); clf
plot(time_peri, mean_off_peri, 'r', 'LineWidth', 1.5)
hold on
plot(time_peri, mean_on_peri, 'b', 'LineWidth', 1.5)

xline(0, 'k--')  % event alignment
xlabel('Time (s)')
ylabel('Z-score')
legend({'ON-OFF', 'OFF-ON'})
title('Mean NE Trace During State Transitions')
box off
exportgraphics(gcf, fullfile('figures','figure4_transitionsNE.jpg'), 'Resolution', 300);
