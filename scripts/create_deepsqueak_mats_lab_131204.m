% create_deepsqueak_mats_lab_131204.m
% Creates DeepSqueak-compatible .mat files from Raven selection tables for the
% lab 131204 batch (London first lab cohort, 17 couples, 25,770 chunks).
%
% Same pipeline as create_deepsqueak_mats_3452.m but pointing to lab data:
%   - Raven tables: raven_tables_lab_131204/ (exported from
%     results/batch_lab_full_softnotch_20260513_1538/detections_clean/)
%   - WAV files:    USV_lab_131204_chunked_2s_full/ (flat directory of 2-s chunk WAVs)
%
% Input membership reflects events_clean.parquet (41,061 events) — the
% <300 ms post-hoc duration filter has already been applied upstream.
% See docs/handoffs/2026-05-14_lab_131204_post_labeling.md.
%
% Full headless pipeline (run in order):
%   1. create_deepsqueak_mats_lab_131204.m   (Raven TSV -> .mat)        <-- this file
%   2. deepsqueak_batch_classify.m           (headless classification)
%   3. deepsqueak_export_stats.m             (export Excel stats)
%   4. import_deepsqueak_results.py          (Python: merge with detections)
%
% Usage:
%   >> run('\\wsl.localhost\Ubuntu\home\shachar\projects\mickey_london_lab\scripts\create_deepsqueak_mats_lab_131204.m')

% --- Configuration ---
ravenDir = '\\wsl.localhost\Ubuntu\home\shachar\projects\mickey_london_lab\raven_tables_lab_131204';
wavDirs  = {'\\wsl.localhost\Ubuntu\home\shachar\projects\mickey_london_lab\USV_lab_131204_chunked_2s_full'};
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

fprintf('\n=== Creating DeepSqueak .mat files for lab 131204 ===\n\n');

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

% --- Next steps ---
fprintf('Next steps (in MATLAB):\n');
fprintf('  1. Run classification:\n');
fprintf('     >> deepsqueak_batch_classify( ...\n');
fprintf('            ''%s'', ...\n', outDir);
fprintf('            ''C:\\path\\to\\DeepSqueak'', ...\n');
fprintf('            ''\\\\wsl.localhost\\Ubuntu\\home\\shachar\\projects\\mickey_london_lab\\deepsqueak_output_lab_131204'', ...\n');
fprintf('            ''kmeans'')\n\n');
fprintf('  2. Then in Python/WSL:\n');
fprintf('     PYTHONPATH=. .venv/bin/python scripts/import_deepsqueak_results.py \\\n');
fprintf('         --results-dir deepsqueak_output_lab_131204 \\\n');
fprintf('         --detections-dir results/batch_lab_full_softnotch_20260513_1538/detections_clean \\\n');
fprintf('         --batch-format \\\n');
fprintf('         --output classified_detections_lab_131204.csv\n\n');
