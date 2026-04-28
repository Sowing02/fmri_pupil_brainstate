clear
RGB = orderedcolors("gem");

% Load Ach data
pupil_Ach_T = readtable("pupil_Ach.csv");
fs_pupil = 30;
pupil = pupil_Ach_T.pupil;
time = pupil_Ach_T.time;
Ach = pupil_Ach_T.Ach;

%% PREPROCESSING
% Remove NaN values
pupil_valid = ~isnan(pupil);
time_valid = ~isnan(time);
Ach_valid = ~isnan(Ach);
valid = pupil_valid & time_valid & Ach_valid;

% Filter valid data based on the valid indices
pupil = pupil(valid);
time = time(valid);
Ach = Ach(valid);

% Crop trace to first 300s
trace_end_s = 300;
trace_end = trace_end_s * fs_pupil;
pupil = pupil(1:trace_end,:);
time = time(1:trace_end,:);
Ach = Ach(1:trace_end,:);

% Normalise data
pupil_z = zscore(pupil);
Ach_z = zscore(Ach);

%% VISUALISATIONS: raw data
figure(1); clf

yyaxis left
plot(time, pupil,"Color",[0.5 0.5 0.5], "LineWidth",0.7)
ax = gca;
ax.YColor = 'k'; 
ylabel('Pupil size (a.u.)')

yyaxis right
plot(time, Ach, 'Color', RGB(2,:), 'LineWidth',0.7)
ax = gca;
ax.YColor = RGB(2,:);
ylabel('ACh signal (\DeltaF/F)')

xlabel('Time (s)')
title('Raw signals (Ach)')
legend('Pupil', 'ACh')
box off

%% ANALYSIS: overall correlation
[R, P] = corrcoef(pupil_z, Ach_z);

r = R(1,2);
p = P(1,2);

% Display results
fprintf('Correlation (r) = %.4f\n', r);
fprintf('p-value = %.4e\n', p);

%% ANALYSIS: t-test for overall correlation

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
save("Ach_data.mat", 'pupil_z', 'time', 'Ach_z', 'fs_pupil','RGB','trace_end_s','trace_end');
save("Ach_results.mat", 'r', 't', 'df', 'p_overall_corr');

exportgraphics(gcf, fullfile('figures','figure1_rawAch.jpg'), 'Resolution', 300);