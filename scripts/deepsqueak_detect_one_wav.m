% deepsqueak_detect_one_wav.m
% Runs DeepSqueak's OWN detector on a SINGLE WAV file, for the
% bridge-vs-native classification experiment (see
% docs/handoffs/2026-05-17_deepsqueak_classification_audit.md).
%
% This is a focused single-file variant of deepsqueak_detect_independent.m.
% It exists so the audit experiment can run without editing the larger
% script. All variables are local to this file; nothing in the workspace
% has to pre-exist.
%
% Pipeline this script feeds into:
%   1. deepsqueak_detect_one_wav.m     <-- this file (MATLAB)
%   2. deepsqueak_batch_classify(...)  (MATLAB, k-means on contour features)
%   3. deepsqueak_export_stats(...)    (MATLAB, 18 features per call to .xlsx)
%   4. (Python) compare features at the target call timestamp against
%      classified_detections_full.csv (the current bridge output)
%
% Usage:
%   >> run('\\wsl.localhost\Ubuntu\home\shachar\projects\mickey_london_lab\scripts\deepsqueak_detect_one_wav.m')
%
% To swap the target file, change TARGET_STEM below.

%% === Configuration ===

% Slide-14 audit target (Complex slot-01, sin=5.69 on a clean arch).
% Other audit-flagged stems you can swap in:
%   '2024-09-30_11-18-27_0000003'  (Up    slot-01, slope=+478, looks flat)
%   '2024-09-30_11-19-38_0000015'  (Chev  slot-01, downward squiggle)
%   '2024-09-30_11-20-14_0000022'  (Cplx  slot-01, clean arch)  <-- default
%   '2024-09-30_11-21-01_0000034'  (FJ    slot-01, no clean step)
TARGET_STEM = '2024-09-30_11-20-14_0000022';

% Cohort directory holding the WAV. Note the SPACE in '5970 USV'.
WAV_DIR = '\\wsl.localhost\Ubuntu\home\shachar\projects\mickey_london_lab\5970 USV';

% Dedicated output dir so we don't collide with the 200-file validation run
% that lives in results/deepsqueak_independent.
outputDir = '\\wsl.localhost\Ubuntu\home\shachar\projects\mickey_london_lab\results\deepsqueak_bridge_test_22';

% DeepSqueak detection parameters (same as deepsqueak_detect_independent.m)
FREQ_LOW_KHZ    = 25;
FREQ_HIGH_KHZ   = 125;
SCORE_THRESHOLD = 0;     % keep everything; filter in Python later

%% === Setup ===

fprintf('\n');
fprintf('===========================================================\n');
fprintf(' DeepSqueak Single-WAV Detection (bridge experiment)\n');
fprintf('===========================================================\n');
fprintf('  Target stem:  %s\n', TARGET_STEM);
fprintf('  WAV dir:      %s\n', WAV_DIR);
fprintf('  Output dir:   %s\n', outputDir);
fprintf('  Freq range:   %d-%d kHz\n', FREQ_LOW_KHZ, FREQ_HIGH_KHZ);
fprintf('  Score thresh: %.2f\n', SCORE_THRESHOLD);
fprintf('===========================================================\n\n');

% DeepSqueak path
dsFolder = fileparts(which('DeepSqueak'));
if isempty(dsFolder)
    error(['DeepSqueak not found on MATLAB path. ' ...
           'Add it with: addpath(genpath(''C:\\path\\to\\DeepSqueak''))']);
end
addpath(genpath(fullfile(dsFolder, 'Functions')));

% Verify the target WAV exists
wavPath = fullfile(WAV_DIR, [TARGET_STEM '.wav']);
if ~isfile(wavPath)
    error('Target WAV not found: %s', wavPath);
end
fprintf('Target WAV: %s\n', wavPath);

% Output subfolders
detectionsOutDir = fullfile(outputDir, 'mat_files');
if ~isfolder(detectionsOutDir)
    mkdir(detectionsOutDir);
end

%% === Network selection (prefer mouse-specific) ===

networkDir = fullfile(dsFolder, 'Networks');
netFiles = dir(fullfile(networkDir, '*.mat'));
if isempty(netFiles)
    error('No detection networks found in: %s', networkDir);
end

selectedNet = '';
for n = 1:length(netFiles)
    nameLower = lower(netFiles(n).name);
    if contains(nameLower, 'mouse') && ~contains(nameLower, 'audible')
        selectedNet = fullfile(networkDir, netFiles(n).name);
        break;
    end
end
if isempty(selectedNet)
    selectedNet = fullfile(networkDir, netFiles(1).name);
    fprintf('[WARN] No mouse-specific network found. Using: %s\n', netFiles(1).name);
end
fprintf('Network: %s\n', selectedNet);

NeuralNetwork = load(selectedNet);

%% === Detection ===

set(0, 'DefaultFigureVisible', 'off');

Settings = [0, FREQ_HIGH_KHZ, FREQ_LOW_KHZ, SCORE_THRESHOLD];
fprintf('\nRunning SqueakDetect ... ');
tStart = tic;

try
    Calls = SqueakDetect(wavPath, NeuralNetwork, TARGET_STEM, Settings, ...
                         1, 1, selectedNet);
catch ME
    set(0, 'DefaultFigureVisible', 'on');
    fprintf('[FAIL]\n');
    fprintf('Error: %s\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('  at %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    end
    rethrow(ME);
end

elapsed = toc(tStart);
set(0, 'DefaultFigureVisible', 'on');

if isempty(Calls) || (istable(Calls) && height(Calls) == 0)
    fprintf('0 calls (%.1fs)\n', elapsed);
    fprintf('[ABORT] DeepSqueak found no calls in this WAV. Nothing to save.\n');
    return;
end

nCalls = height(Calls);
fprintf('%d calls (%.1fs)\n', nCalls, elapsed);

%% === Save .mat (UNC-safe via tempdir copy) ===

audiodata = audioinfo(wavPath);
matOutPath = fullfile(detectionsOutDir, [TARGET_STEM '.mat']);
tmpMat = fullfile(tempdir, [TARGET_STEM '.mat']);
save(tmpMat, 'Calls', 'audiodata', '-v7.3');
copyfile(tmpMat, matOutPath);
delete(tmpMat);
fprintf('Saved: %s\n', matOutPath);

%% === Write a small CSV with the detections ===

beginTimes = zeros(nCalls, 1);
endTimes   = zeros(nCalls, 1);
durations  = zeros(nCalls, 1);
lowFreqs   = zeros(nCalls, 1);
highFreqs  = zeros(nCalls, 1);
scores     = zeros(nCalls, 1);
hasScore = ismember('Score', Calls.Properties.VariableNames);

for c = 1:nCalls
    box = Calls.Box(c, :);
    beginTimes(c) = box(1);
    endTimes(c)   = box(1) + box(3);
    durations(c)  = box(3);
    lowFreqs(c)   = box(2);
    highFreqs(c)  = box(2) + box(4);
    if hasScore
        scores(c) = Calls.Score(c);
    end
end

T = table(repmat({TARGET_STEM}, nCalls, 1), beginTimes, endTimes, durations, ...
          lowFreqs, highFreqs, scores, ...
    'VariableNames', {'wav_stem', 'begin_time_s', 'end_time_s', ...
                      'duration_s', 'low_freq_khz', 'high_freq_khz', 'score'});

csvPath = fullfile(outputDir, 'deepsqueak_one_wav_detections.csv');
tmpCsv = fullfile(tempdir, 'deepsqueak_one_wav_detections.csv');
writetable(T, tmpCsv);
copyfile(tmpCsv, csvPath);
delete(tmpCsv);
fprintf('Wrote: %s\n', csvPath);

%% === Summary + next-step hint ===

fprintf('\n--- Summary ---\n');
fprintf('  Calls detected: %d\n', nCalls);
fprintf('  Mat file:       %s\n', matOutPath);
fprintf('  CSV:            %s\n', csvPath);

% Print rows whose begin_time is near the audit-target call time (5.992s).
% Tolerance = 0.075 s matches import_deepsqueak_results.py --batch-format,
% so "found by DS" here means the same thing it means in the bridge.
auditTime = 5.992;
matchTolSec = 0.075;
near = abs(beginTimes - auditTime) < matchTolSec;
if any(near)
    fprintf('\nDetections within +/-%d ms of audit-target time (%.3fs):\n', ...
            round(matchTolSec * 1000), auditTime);
    disp(T(near, :));
else
    fprintf(['\n[NOTE] No native detection within +/-%d ms of %.3fs ' ...
             '(the audit-flagged call).\n'], round(matchTolSec * 1000), auditTime);
    fprintf('       DS''s own detector may have missed it; widen the window if needed.\n');
end

fprintf('\nNext step (MATLAB):\n');
fprintf('  >> deepsqueak_batch_classify(''%s'', ''%s'', ''%s'', ''kmeans'')\n', ...
    strrep(detectionsOutDir, '\', '\\'), ...
    strrep(dsFolder, '\', '\\'), ...
    strrep(fullfile(outputDir, 'classified'), '\', '\\'));
fprintf('\n');
