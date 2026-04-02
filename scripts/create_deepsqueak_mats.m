% create_deepsqueak_mats.m (v2)
% Creates DeepSqueak-compatible .mat files from Raven selection tables.
%
% Fixes over v1:
%   - Explicit 'VariableNames' in table constructor (matches DeepSqueak
%     import_raven_Callback.m:49 exactly)
%   - Verify-after-save: reloads each .mat and checks structure
%   - Prints Box sample for visual confirmation
%
% Full headless pipeline (run in order):
%   1. create_deepsqueak_mats.m     (Raven TSV -> .mat)       <-- this file
%   2. deepsqueak_batch_classify.m  (headless classification)
%   3. deepsqueak_export_stats.m    (export Excel stats)
%   4. import_deepsqueak_results.py (Python: merge with detections)
%
% Usage:
%   >> run('\\wsl.localhost\Ubuntu\home\shachar\projects\mickey_london_lab\scripts\create_deepsqueak_mats.m')

% --- Configuration ---
% Switch between smoke test (10 files) and full run (1328 files):
%   ravenDir: raven_tables (smoke) or raven_tables_full (full)
%   wavDirs:  list of root directories to search recursively for WAVs

ravenDir = '\\wsl.localhost\Ubuntu\home\shachar\projects\mickey_london_lab\raven_tables_full';
wavDirs  = {'\\wsl.localhost\Ubuntu\home\shachar\projects\mickey_london_lab\5970', ...
             '\\wsl.localhost\Ubuntu\home\shachar\projects\mickey_london_lab\5970_reviewed'};
outDir   = fullfile(fileparts(which('DeepSqueak')), 'Detections');

% Build WAV lookup: stem -> full path (recursive search across all wavDirs)
fprintf('Building WAV lookup table...\n');
wavLookup = containers.Map();
for d = 1:length(wavDirs)
    wavFiles_d = dir(fullfile(wavDirs{d}, '**', '*.wav'));
    for w = 1:length(wavFiles_d)
        [~, stem, ~] = fileparts(wavFiles_d(w).name);
        wavLookup(stem) = fullfile(wavFiles_d(w).folder, wavFiles_d(w).name);
    end
end
fprintf('Found %d WAV files across %d directories.\n\n', wavLookup.Count, length(wavDirs));

if ~isfolder(outDir)
    mkdir(outDir);
    fprintf('Created output directory: %s\n', outDir);
end

ravenFiles = dir(fullfile(ravenDir, '*.txt'));
totalSaved = 0;
totalFailed = 0;

fprintf('\n=== Creating DeepSqueak .mat files (v2) ===\n\n');

for i = 1:length(ravenFiles)
    % Skip non-Raven files
    if strcmp(ravenFiles(i).name, 'export_summary.json')
        continue;
    end

    ravenPath = fullfile(ravenDir, ravenFiles(i).name);

    % Extract WAV stem from filename: STEM.Table.1.selections.txt
    tokens = regexp(ravenFiles(i).name, '(.+)\.Table\.1\.selections\.txt', 'tokens');
    if isempty(tokens)
        fprintf('[SKIP] Cannot parse filename: %s\n', ravenFiles(i).name);
        continue;
    end
    wavStem = tokens{1}{1};

    if ~wavLookup.isKey(wavStem)
        fprintf('[SKIP] No WAV file for: %s\n', ravenFiles(i).name);
        continue;
    end
    wavPath = wavLookup(wavStem);

    % Read Raven selection table
    ravenTable = readtable(ravenPath, 'Delimiter', 'tab');

    % Compute DeltaTime if not present
    if ~ismember('DeltaTime_s_', ravenTable.Properties.VariableNames)
        ravenTable.DeltaTime_s_ = ravenTable.EndTime_s_ - ravenTable.BeginTime_s_;
    end

    nCalls = height(ravenTable);
    fprintf('--- %s (%d calls) ---\n', wavStem, nCalls);

    % Build Box: [BeginTime_s, LowFreq_kHz, DeltaTime_s, Bandwidth_kHz]
    % This matches DeepSqueak's import_raven_Callback.m box layout
    Box = [ravenTable.BeginTime_s_, ...
           ravenTable.LowFreq_Hz_ / 1000, ...
           ravenTable.DeltaTime_s_, ...
           (ravenTable.HighFreq_Hz_ - ravenTable.LowFreq_Hz_) / 1000];

    Score  = ones(nCalls, 1);
    Accept = ones(nCalls, 1);
    Type   = categorical(repmat({'USV'}, nCalls, 1));
    Power  = zeros(nCalls, 1);

    % Build Calls table with EXPLICIT VariableNames
    % (matches DeepSqueak import_raven_Callback.m:49)
    Calls = table(Box, Score, Accept, Type, Power, ...
        'VariableNames', {'Box', 'Score', 'Accept', 'Type', 'Power'});

    % Get audio metadata (struct with Filename, Duration, SampleRate, etc.)
    audiodata = audioinfo(wavPath);

    % Save as v7.3 MAT (HDF5) — required by DeepSqueak
    outPath = fullfile(outDir, [wavStem '.mat']);
    save(outPath, 'Calls', 'audiodata', '-v7.3');
    fprintf('  Saved -> %s\n', outPath);

    % --- Verify after save ---
    verify = load(outPath);

    verifyOK = true;

    % Check 1: Calls exists and is a table
    if ~isfield(verify, 'Calls') || ~istable(verify.Calls)
        fprintf('  [VERIFY FAIL] Calls not a table after reload!\n');
        verifyOK = false;
    end

    % Check 2: Correct number of calls survived save/load
    if verifyOK && height(verify.Calls) ~= nCalls
        fprintf('  [VERIFY FAIL] Expected %d calls, got %d after reload!\n', ...
            nCalls, height(verify.Calls));
        verifyOK = false;
    end

    % Check 3: Variable names preserved
    if verifyOK
        actualVars = verify.Calls.Properties.VariableNames;
        expectedVars = {'Box', 'Score', 'Accept', 'Type', 'Power'};
        if ~isequal(actualVars, expectedVars)
            fprintf('  [VERIFY FAIL] VariableNames mismatch after reload!\n');
            fprintf('    Expected: %s\n', strjoin(expectedVars, ', '));
            fprintf('    Got:      %s\n', strjoin(actualVars, ', '));
            verifyOK = false;
        end
    end

    % Check 4: audiodata.Filename resolves
    if verifyOK && isfield(verify.audiodata, 'Filename')
        if ~isfile(verify.audiodata.Filename)
            fprintf('  [VERIFY WARN] audiodata.Filename does not resolve: %s\n', ...
                verify.audiodata.Filename);
            fprintf('    DeepSqueak may prompt for audio location.\n');
        end
    end

    % Check 5: No zero-dimension boxes
    if verifyOK && height(verify.Calls) > 0
        zeroW = sum(verify.Calls.Box(:,3) == 0);
        zeroH = sum(verify.Calls.Box(:,4) == 0);
        if zeroW > 0 || zeroH > 0
            fprintf('  [VERIFY WARN] %d zero-width + %d zero-height boxes\n', zeroW, zeroH);
        end
    end

    if verifyOK
        fprintf('  [VERIFY OK] %d calls, names match, structure correct\n', nCalls);
        totalSaved = totalSaved + 1;
    else
        totalFailed = totalFailed + 1;
    end

    % Print first call's Box for visual spot-check
    if nCalls > 0
        fprintf('  Sample Box[1]: Begin=%.4fs  LowFreq=%.1fkHz  Duration=%.4fs  BW=%.1fkHz\n', ...
            verify.Calls.Box(1,1), verify.Calls.Box(1,2), ...
            verify.Calls.Box(1,3), verify.Calls.Box(1,4));
    end

    fprintf('\n');
end

fprintf('=== Done: %d saved, %d failed ===\n', totalSaved, totalFailed);
if totalFailed == 0
    fprintf('Open DeepSqueak -> File -> Load Calls to test.\n');
else
    fprintf('Check VERIFY FAIL messages above before loading in DeepSqueak.\n');
end
fprintf('\n');
