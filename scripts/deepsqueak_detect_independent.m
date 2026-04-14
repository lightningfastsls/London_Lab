% deepsqueak_detect_independent.m
% Runs DeepSqueak's OWN detection on raw WAV files — independent of our CNN.
%
% Purpose:
%   Cross-validate our CNN detection pipeline. Our CNN found 7,575 USVs in
%   cage 5970. This script lets DeepSqueak find USVs independently, so we
%   can compare two completely different detectors. High agreement = strong
%   evidence the detections are real.
%
% What this does NOT do:
%   - Does NOT import our CNN detections (that's create_deepsqueak_mats.m)
%   - Does NOT use our Raven selection tables
%   - DeepSqueak runs its own neural network on raw audio from scratch
%
% Output:
%   - .mat files in outputDir (standard DeepSqueak detection format)
%   - deepsqueak_independent_detections.csv (for Python comparison script)
%
% Prerequisites:
%   - DeepSqueak v3.x installed and on MATLAB path
%   - A detection network in DeepSqueak/Networks/ (e.g., 'Mouse Call Detection Network')
%   - WAV files accessible from MATLAB
%
% Full cross-validation pipeline:
%   1. deepsqueak_detect_independent.m    <-- this file (MATLAB)
%   2. compare_detections.py              (Python: compare CNN vs DS)
%
% Usage:
%   >> run('\\wsl.localhost\Ubuntu\home\shachar\projects\mickey_london_lab\scripts\deepsqueak_detect_independent.m')
%
%   For API verification only (no detection):
%   >> VERIFY_ONLY = true;
%   >> run('...\deepsqueak_detect_independent.m')

%% === Configuration ===

% WAV directories to search (same as batch detection)
wavDirs = {'\\wsl.localhost\Ubuntu\home\shachar\projects\mickey_london_lab\5970', ...
           '\\wsl.localhost\Ubuntu\home\shachar\projects\mickey_london_lab\5970_reviewed'};

% DeepSqueak installation
dsFolder = fileparts(which('DeepSqueak'));
if isempty(dsFolder)
    error('DeepSqueak not found on MATLAB path. Add it with: addpath(''C:\\path\\to\\DeepSqueak'')');
end

% Output directory (separate from our CNN detections!)
outputDir = '\\wsl.localhost\Ubuntu\home\shachar\projects\mickey_london_lab\results\deepsqueak_independent';

% Detection network — list available networks and pick one
networkDir = fullfile(dsFolder, 'Networks');

% Detection parameters
FREQ_LOW_KHZ   = 25;    % Lower frequency bound (kHz) — mouse USVs start ~25 kHz
FREQ_HIGH_KHZ  = 125;   % Upper frequency bound (kHz) — mouse USVs go up to ~125 kHz
SCORE_THRESHOLD = 0;     % Detection confidence threshold (0-1). 0 = keep everything,
                         %   filter later in Python. DeepSqueak default is ~0.
OVERLAP = 0.5;           % Detection window overlap fraction

% Sample control
MAX_FILES = 200;     % Max WAV files to process (0 = all). 200 is a good validation sample.
RANDOM_SEED = 42;    % For reproducible sampling

% Verify-only mode (set to true to check API without running detection)
if ~exist('VERIFY_ONLY', 'var')
    VERIFY_ONLY = false;
end

%% === Setup ===

fprintf('\n');
fprintf('===========================================================\n');
fprintf(' DeepSqueak INDEPENDENT Detection — Cross-Validation\n');
fprintf('===========================================================\n');
fprintf('  DS folder:    %s\n', dsFolder);
fprintf('  Freq range:   %d-%d kHz\n', FREQ_LOW_KHZ, FREQ_HIGH_KHZ);
fprintf('  Threshold:    %.2f\n', SCORE_THRESHOLD);
fprintf('  Max files:    %d\n', MAX_FILES);
fprintf('  Verify only:  %s\n', mat2str(VERIFY_ONLY));
fprintf('===========================================================\n\n');

% Add DeepSqueak functions to path
dsFunc = fullfile(dsFolder, 'Functions');
if ~isfolder(dsFunc)
    error('DeepSqueak Functions/ folder not found at: %s', dsFunc);
end
addpath(genpath(dsFunc));
fprintf('Added DeepSqueak functions to path.\n');

%% === Step 0: API Discovery & Verification ===

fprintf('\n--- Step 0: API Discovery ---\n\n');

% Check what detection networks are available
fprintf('Available detection networks:\n');
if isfolder(networkDir)
    netFiles = dir(fullfile(networkDir, '*.mat'));
    for n = 1:length(netFiles)
        fprintf('  [%d] %s\n', n, netFiles(n).name);
    end
    if isempty(netFiles)
        fprintf('  [NONE] No .mat files in Networks/ folder!\n');
        error('No detection networks found in: %s', networkDir);
    end
else
    error('Networks folder not found: %s', networkDir);
end
fprintf('\n');

% Check if SqueakDetect exists
if exist('SqueakDetect', 'file') == 2
    fprintf('[OK] SqueakDetect found: %s\n', which('SqueakDetect'));
    try
        nArgs = nargin('SqueakDetect');
        fprintf('[OK] SqueakDetect accepts %d arguments\n', nArgs);
    catch
        fprintf('[INFO] Could not determine nargin (may use varargin)\n');
    end
else
    fprintf('[WARN] SqueakDetect not found on path!\n');
    fprintf('       Detection may use a different function name in this version.\n');
    % Try to find alternative detection functions
    detFuncs = {'SqueakDetect', 'squeakDetect', 'detect_calls', 'DetectCalls'};
    for df = 1:length(detFuncs)
        w = which(detFuncs{df});
        if ~isempty(w)
            fprintf('  Found alternative: %s -> %s\n', detFuncs{df}, w);
        end
    end
end

% Check loadCallfile (we'll need it for reading results)
if exist('loadCallfile', 'file') == 2
    fprintf('[OK] loadCallfile found: %s\n', which('loadCallfile'));
else
    fprintf('[FAIL] loadCallfile not found — DeepSqueak functions not properly loaded\n');
end

% Load first available detection network to inspect its structure
fprintf('\nInspecting first detection network structure...\n');
networkPath = fullfile(networkDir, netFiles(1).name);
networkData = load(networkPath);
netFields = fieldnames(networkData);
fprintf('  File: %s\n', netFiles(1).name);
fprintf('  Fields: ');
fprintf('%s ', netFields{:});
fprintf('\n');

% Identify the network variable (usually 'detector' or 'net' or 'NeuralNetwork')
networkVarNames = {'detector', 'net', 'NeuralNetwork', 'network', 'Network'};
detectorVar = '';
for vn = 1:length(networkVarNames)
    if isfield(networkData, networkVarNames{vn})
        detectorVar = networkVarNames{vn};
        detectorObj = networkData.(detectorVar);
        fprintf('  Network variable: %s (class: %s)\n', detectorVar, class(detectorObj));
        break;
    end
end
if isempty(detectorVar)
    fprintf('  [WARN] Could not identify network variable among known names.\n');
    fprintf('         Known fields: %s\n', strjoin(netFields, ', '));
end

if VERIFY_ONLY
    fprintf('\n=== VERIFY_ONLY mode — stopping here. ===\n');
    fprintf('Review the output above, then set VERIFY_ONLY = false and re-run.\n\n');
    return;
end

%% === Step 1: Select network ===

fprintf('\n--- Step 1: Network Selection ---\n\n');

% Auto-select: prefer a mouse-specific network
selectedNet = '';
for n = 1:length(netFiles)
    name_lower = lower(netFiles(n).name);
    if contains(name_lower, 'mouse') && ~contains(name_lower, 'audible')
        selectedNet = fullfile(networkDir, netFiles(n).name);
        fprintf('Auto-selected: %s\n', netFiles(n).name);
        break;
    end
end
if isempty(selectedNet)
    % Fallback: use the first network
    selectedNet = fullfile(networkDir, netFiles(1).name);
    fprintf('No mouse-specific network found. Using: %s\n', netFiles(1).name);
end

% Load the selected network
NeuralNetwork = load(selectedNet);
fprintf('Network loaded.\n\n');

%% === Step 2: Build WAV file list ===

fprintf('--- Step 2: Building WAV file list ---\n\n');

allWavPaths = {};
for d = 1:length(wavDirs)
    wavFiles_d = dir(fullfile(wavDirs{d}, '**', '*.wav'));
    for w = 1:length(wavFiles_d)
        allWavPaths{end+1} = fullfile(wavFiles_d(w).folder, wavFiles_d(w).name); %#ok<AGROW>
    end
end
fprintf('Found %d total WAV files across %d directories.\n', length(allWavPaths), length(wavDirs));

% Sample if needed
if MAX_FILES > 0 && MAX_FILES < length(allWavPaths)
    rng(RANDOM_SEED);
    idx = randperm(length(allWavPaths), MAX_FILES);
    allWavPaths = allWavPaths(sort(idx));
    fprintf('Sampled %d files (seed=%d).\n', MAX_FILES, RANDOM_SEED);
end
fprintf('\n');

%% === Step 3: Create output directory ===

if ~isfolder(outputDir)
    mkdir(outputDir);
    fprintf('Created output directory: %s\n\n', outputDir);
end

% Create detections subfolder
detectionsOutDir = fullfile(outputDir, 'mat_files');
if ~isfolder(detectionsOutDir)
    mkdir(detectionsOutDir);
end

%% === Step 4: Run detection ===

fprintf('--- Step 3: Running DeepSqueak Detection ---\n\n');
fprintf('Processing %d files...\n\n', length(allWavPaths));

% Build Settings as a numeric array (DeepSqueak indexes by position):
%   Settings(1) = detection length in seconds (0 = full file)
%   Settings(2) = high frequency cutoff (kHz)
%   Settings(3) = low frequency cutoff (kHz)
%   Settings(4) = score threshold (0-1)
Settings = [0, FREQ_HIGH_KHZ, FREQ_LOW_KHZ, SCORE_THRESHOLD];

% Suppress figure display
set(0, 'DefaultFigureVisible', 'off');

% Results accumulator for CSV export
csvRows = {};  % {wav_stem, begin_time_s, end_time_s, duration_s, low_freq_khz, high_freq_khz, score}

totalDetections = 0;
filesWithDetections = 0;
failedFiles = 0;
processTimes = [];

for i = 1:length(allWavPaths)
    wavPath = allWavPaths{i};
    [~, wavStem, ~] = fileparts(wavPath);

    fprintf('  [%d/%d] %s ... ', i, length(allWavPaths), wavStem);
    tStart = tic;

    try
        % ============================================================
        % CORE DETECTION CALL
        % ============================================================
        % SqueakDetect signature (from source, line 1):
        %   SqueakDetect(inputfile, networkfile, fname, Settings, currentFile, totalFiles, networkname)
        %
        % Args:
        %   inputfile   = WAV path (string)
        %   networkfile = loaded network struct (with .detector, .wind, .noverlap, .nfft, .imLength)
        %   fname       = display name for progress messages (string)
        %   Settings    = [detect_length_s, high_freq_khz, low_freq_khz, score_threshold]
        %   currentFile = file index (integer)
        %   totalFiles  = total file count (integer)
        %   networkname = network file path (string)
        % ============================================================

        Calls = SqueakDetect(wavPath, NeuralNetwork, wavStem, Settings, ...
            i, length(allWavPaths), selectedNet);

        elapsed = toc(tStart);
        processTimes(end+1) = elapsed; %#ok<SAGROW>

        if isempty(Calls) || (istable(Calls) && height(Calls) == 0)
            fprintf('0 calls (%.1fs)\n', elapsed);
        else
            nCalls = height(Calls);
            totalDetections = totalDetections + nCalls;
            filesWithDetections = filesWithDetections + 1;
            fprintf('%d calls (%.1fs)\n', nCalls, elapsed);

            % Save .mat file (use temp+copy for UNC paths)
            audiodata = audioinfo(wavPath);
            matOutPath = fullfile(detectionsOutDir, [wavStem '.mat']);
            tmpPath = fullfile(tempdir, [wavStem '.mat']);
            save(tmpPath, 'Calls', 'audiodata', '-v7.3');
            copyfile(tmpPath, matOutPath);
            delete(tmpPath);

            % Accumulate CSV rows
            for c = 1:nCalls
                box = Calls.Box(c,:);
                score_val = 0;
                if ismember('Score', Calls.Properties.VariableNames)
                    score_val = Calls.Score(c);
                end
                csvRows{end+1} = {wavStem, ...
                    box(1), ...                    % begin_time_s
                    box(1) + box(3), ...           % end_time_s
                    box(3), ...                    % duration_s
                    box(2), ...                    % low_freq_khz
                    box(2) + box(4), ...           % high_freq_khz
                    score_val}; %#ok<AGROW>
            end
        end

    catch ME
        elapsed = toc(tStart);
        failedFiles = failedFiles + 1;
        fprintf('[ERROR] (%.1fs) %s\n', elapsed, ME.message);

        % On first failure, print more detail to help debug API issues
        if failedFiles == 1
            fprintf('\n    === FIRST FAILURE DIAGNOSTIC ===\n');
            fprintf('    Error ID: %s\n', ME.identifier);
            fprintf('    Full message: %s\n', ME.message);
            if ~isempty(ME.stack)
                fprintf('    At: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
            end
            fprintf('    \n');
            fprintf('    If this is an argument count error, try:\n');
            fprintf('      >> help SqueakDetect\n');
            fprintf('      >> nargin(''SqueakDetect'')\n');
            fprintf('    Then adjust the SqueakDetect call in this script.\n');
            fprintf('    === END DIAGNOSTIC ===\n\n');
        end

        % Bail out after 5 consecutive failures (likely API mismatch)
        if failedFiles >= 5
            fprintf('\n[ABORT] 5+ failures — likely a SqueakDetect API mismatch.\n');
            fprintf('Run with VERIFY_ONLY=true to inspect the API, then fix the call.\n');
            break;
        end
    end
end

% Restore figure visibility
set(0, 'DefaultFigureVisible', 'on');

%% === Step 5: Export CSV ===

fprintf('\n--- Step 4: Exporting results ---\n\n');

csvPath = fullfile(outputDir, 'deepsqueak_independent_detections.csv');

if ~isempty(csvRows)
    % Build table
    csvData = cell2mat(cellfun(@(r) [r{2:end}], csvRows, 'UniformOutput', false)');
    csvStems = cellfun(@(r) r{1}, csvRows, 'UniformOutput', false);

    T = table(csvStems(:), csvData(:,1), csvData(:,2), csvData(:,3), ...
              csvData(:,4), csvData(:,5), csvData(:,6), ...
        'VariableNames', {'wav_stem', 'begin_time_s', 'end_time_s', ...
                          'duration_s', 'low_freq_khz', 'high_freq_khz', 'score'});

    % Write CSV (use temp+copy for UNC paths)
    tmpCsv = fullfile(tempdir, 'deepsqueak_independent_detections.csv');
    writetable(T, tmpCsv);
    copyfile(tmpCsv, csvPath);
    delete(tmpCsv);
    fprintf('Wrote %d detections to: %s\n', height(T), csvPath);
else
    fprintf('[WARN] No detections found — CSV not written.\n');
end

% Also write the list of files that were processed (for comparison script)
fileListPath = fullfile(outputDir, 'processed_files.txt');
tmpList = fullfile(tempdir, 'processed_files.txt');
fid = fopen(tmpList, 'w');
for i = 1:length(allWavPaths)
    [~, stem, ~] = fileparts(allWavPaths{i});
    fprintf(fid, '%s\n', stem);
end
fclose(fid);
copyfile(tmpList, fileListPath);
delete(tmpList);
fprintf('Wrote file list (%d files) to: %s\n', length(allWavPaths), fileListPath);

%% === Summary ===

fprintf('\n===========================================================\n');
fprintf(' Detection Complete\n');
fprintf('===========================================================\n');
fprintf('  Files processed:        %d\n', length(allWavPaths));
fprintf('  Files with detections:  %d (%.1f%%)\n', ...
    filesWithDetections, 100*filesWithDetections/max(1,length(allWavPaths)));
fprintf('  Total detections:       %d\n', totalDetections);
fprintf('  Failed files:           %d\n', failedFiles);
if ~isempty(processTimes)
    fprintf('  Avg time per file:      %.1f s\n', mean(processTimes));
    fprintf('  Est. total for 6400:    %.1f hours\n', ...
        mean(processTimes) * 6400 / 3600);
end
fprintf('  Output CSV:             %s\n', csvPath);
fprintf('===========================================================\n');
fprintf('\n  Next step (Python/WSL):\n');
fprintf('    .venv/bin/python scripts/compare_detections.py \\\n');
fprintf('        --cnn-dir results/batch_5970_v2_full/detections \\\n');
fprintf('        --ds-csv %s \\\n', strrep(csvPath, '\', '/'));
fprintf('        --processed-list %s\n\n', strrep(fileListPath, '\', '/'));
