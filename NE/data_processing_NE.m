clear
RGB = orderedcolors("gem");

% Load NE data
pupil_NE_T = readtable("pupil_NE.csv");
fs_pupil = 30;
pupil = pupil_NE_T.pupil;
time = pupil_NE_T.time;
NE = pupil_NE_T.NE;

%% PREPROCESSING
% Remove NaN values
pupil_valid = ~isnan(pupil);
time_valid = ~isnan(time);
NE_valid = ~isnan(NE);
valid = pupil_valid & time_valid & NE_valid;

% Filter valid data based on the valid indices
pupil = pupil(valid);
time = time(valid);
NE = NE(valid);

% Crop trace to first 300s
trace_end_s = 300;
trace_end = trace_end_s * fs_pupil;
pupil = pupil(1:trace_end,:);
time = time(1:trace_end,:);
NE = NE(1:trace_end,:);

% Normalise data
pupil_z = zscore(pupil);
NE_z = zscore(NE);

%% VISUALISATIONS: raw data
figure(1); clf

yyaxis left
plot(time, pupil,"Color",[0.5 0.5 0.5], "LineWidth",0.7)
ax = gca;
ax.YColor = 'k'; 
ylabel('Pupil size (a.u.)')

yyaxis right
plot(time, NE, 'Color', RGB(3,:), 'LineWidth',0.7)
ax = gca;
ax.YColor = RGB(3,:);
ylabel('NE signal (\DeltaF/F)')

xlabel('Time (s)')
title('Raw signals (NE)')
legend('Pupil', 'NE')
xlim([0 300]);
box off

%% ANALYSIS: overall correlation
[R, P] = corrcoef(pupil_z, NE_z);

r = R(1,2);
p = P(1,2);

% Display results
fprintf('Correlation (r) = %.4f\n', r);
fprintf('p-value = %.4e\n', p);

%% ANALYSIS: t-test for correlation

% Number of samples
n = length(pupil_z);

% Compute t-statistic from r
t = r * sqrt((n - 2) / (1 - r^2));

% Degrees of freedom
df = n - 2;

% Two-tailed p-value
p_overall_corr = 2 * (1 - tcdf(abs(t), df));

% Display
fprintf('t-statistic = %.4f\n', t);
fprintf('df = %d\n', df);
fprintf('p-value (t-test) = %.4e\n', p_overall_corr);


% Save data/figures
save("NE_data.mat", 'pupil_z', 'time', 'NE_z', 'fs_pupil','RGB','trace_end_s','trace_end');
save("NE_results.mat", 'r', 't', 'df', 'p_overall_corr');

exportgraphics(gcf, fullfile('figures','figure1_rawNE.jpg'), 'Resolution', 300);
