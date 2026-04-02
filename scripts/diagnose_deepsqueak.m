% diagnose_deepsqueak.m
% Compares our .mat files against DeepSqueak's example to find structural issues.
%
% Usage:
%   1. Open MATLAB
%   2. >> run('\\wsl.localhost\Ubuntu\home\shachar\projects\mickey_london_lab\scripts\diagnose_deepsqueak.m')
%   3. Copy the output and paste it back
%
% What this checks:
%   - Top-level variables match (Calls, audiodata)
%   - Calls is a table (not struct/array)
%   - Column names match DeepSqueak expectations
%   - Box dimensions are non-zero (zero = silently removed)
%   - audiodata.Filename resolves on disk

fprintf('\n======================================\n');
fprintf(' DeepSqueak .mat Diagnostic Report\n');
fprintf('======================================\n\n');

%% --- Config ---
ourDir = fullfile(fileparts(which('DeepSqueak')), 'Detections');
exampleDir = fullfile(fileparts(which('DeepSqueak')), 'Audio');
exampleMat = fullfile(exampleDir, 'Example Mouse Recording.mat');

% Expected column names (from DeepSqueak import_raven_Callback.m:49)
expectedVars = {'Box', 'Score', 'Accept', 'Type', 'Power'};

passCount = 0;
failCount = 0;
warnCount = 0;

%% === PART 1: Inspect DeepSqueak example file ===
fprintf('--- PART 1: DeepSqueak Example File ---\n\n');

if isfile(exampleMat)
    fprintf('[INFO] Found example: %s\n', exampleMat);
    exData = load(exampleMat);

    fnames = fieldnames(exData);
    fprintf('[INFO] Top-level variables: ');
    fprintf('%s ', fnames{:});
    fprintf('\n');

    if isfield(exData, 'Calls')
        fprintf('[INFO] class(Calls) = %s\n', class(exData.Calls));
        fprintf('[INFO] size(Calls) = [%d, %d]\n', size(exData.Calls));
        fprintf('[INFO] VariableNames: ');
        fprintf('%s ', exData.Calls.Properties.VariableNames{:});
        fprintf('\n');
        if height(exData.Calls) > 0
            fprintf('[INFO] First row Box: [%.4f, %.4f, %.4f, %.4f]\n', ...
                exData.Calls.Box(1,:));
        end
    else
        fprintf('[WARN] Example has no "Calls" variable!\n');
    end

    if isfield(exData, 'audiodata')
        exAudioFields = fieldnames(exData.audiodata);
        fprintf('[INFO] audiodata fields: ');
        fprintf('%s ', exAudioFields{:});
        fprintf('\n');
        if isfield(exData.audiodata, 'Filename')
            fprintf('[INFO] audiodata.Filename = %s\n', exData.audiodata.Filename);
        end
    end
    fprintf('\n');
else
    fprintf('[WARN] Example file not found: %s\n', exampleMat);
    fprintf('[WARN] Looked in DeepSqueak/Audio/ folder.\n');
    fprintf('[HINT] If example is elsewhere, edit exampleMat path above.\n\n');
end

%% === PART 2: Find and check our .mat files ===
fprintf('--- PART 2: Our Generated .mat Files ---\n\n');

ourFiles = dir(fullfile(ourDir, '2024-09-30_*.mat'));
if isempty(ourFiles)
    fprintf('[FAIL] No .mat files found in: %s\n', ourDir);
    fprintf('[HINT] Run create_deepsqueak_mats.m first.\n');
    failCount = failCount + 1;
else
    fprintf('[INFO] Found %d .mat files in %s\n\n', length(ourFiles), ourDir);
end

for i = 1:length(ourFiles)
    matPath = fullfile(ourDir, ourFiles(i).name);
    fprintf('=== File %d: %s ===\n', i, ourFiles(i).name);

    data = load(matPath);

    % Check 1: Top-level variables
    vars = fieldnames(data);
    fprintf('  Variables: ');
    fprintf('%s ', vars{:});
    fprintf('\n');

    hasCalls = isfield(data, 'Calls');
    hasAudio = isfield(data, 'audiodata');

    if hasCalls && hasAudio
        fprintf('  [PASS] Has both Calls and audiodata\n');
        passCount = passCount + 1;
    else
        if ~hasCalls, fprintf('  [FAIL] Missing "Calls"\n'); failCount = failCount + 1; end
        if ~hasAudio, fprintf('  [FAIL] Missing "audiodata"\n'); failCount = failCount + 1; end
        fprintf('\n');
        continue;
    end

    % Check 2: Calls is a table
    callsClass = class(data.Calls);
    if strcmp(callsClass, 'table')
        fprintf('  [PASS] Calls is a table\n');
        passCount = passCount + 1;
    else
        fprintf('  [FAIL] Calls is %s (expected table)\n', callsClass);
        failCount = failCount + 1;
        fprintf('\n');
        continue;
    end

    % Check 3: Variable names match
    actualVars = data.Calls.Properties.VariableNames;
    fprintf('  VariableNames: ');
    fprintf('%s ', actualVars{:});
    fprintf('\n');

    if isequal(actualVars, expectedVars)
        fprintf('  [PASS] VariableNames match DeepSqueak expectation\n');
        passCount = passCount + 1;
    else
        fprintf('  [FAIL] VariableNames mismatch!\n');
        fprintf('         Expected: ');
        fprintf('%s ', expectedVars{:});
        fprintf('\n');
        failCount = failCount + 1;
    end

    % Check 4: Call count and Box dimensions
    nCalls = height(data.Calls);
    fprintf('  Call count: %d\n', nCalls);

    if nCalls == 0
        fprintf('  [FAIL] Zero calls in table!\n');
        failCount = failCount + 1;
    else
        fprintf('  [PASS] Has %d calls\n', nCalls);
        passCount = passCount + 1;

        % Check for zero-width or zero-height boxes (loadCallfile removes these)
        zeroWidth = sum(data.Calls.Box(:,3) == 0);
        zeroHeight = sum(data.Calls.Box(:,4) == 0);
        if zeroWidth > 0 || zeroHeight > 0
            fprintf('  [FAIL] %d calls with zero width, %d with zero height\n', ...
                zeroWidth, zeroHeight);
            fprintf('         DeepSqueak silently removes these!\n');
            failCount = failCount + 1;
        else
            fprintf('  [PASS] No zero-dimension boxes\n');
            passCount = passCount + 1;
        end

        % Print first 3 rows of Box
        nShow = min(3, nCalls);
        fprintf('  First %d Box rows:\n', nShow);
        for j = 1:nShow
            fprintf('    [%d] BeginTime=%.4f  LowFreqKHz=%.2f  Duration=%.4f  BandwidthKHz=%.2f\n', ...
                j, data.Calls.Box(j,1), data.Calls.Box(j,2), ...
                data.Calls.Box(j,3), data.Calls.Box(j,4));
        end
    end

    % Check 5: audiodata structure
    audioFields = fieldnames(data.audiodata);
    fprintf('  audiodata fields: ');
    fprintf('%s ', audioFields{:});
    fprintf('\n');

    if isfield(data.audiodata, 'Filename')
        fname = data.audiodata.Filename;
        fprintf('  audiodata.Filename = %s\n', fname);

        if isfile(fname)
            fprintf('  [PASS] audiodata.Filename resolves on disk\n');
            passCount = passCount + 1;
        else
            fprintf('  [FAIL] audiodata.Filename does NOT resolve on disk!\n');
            fprintf('         This triggers DeepSqueak audio re-discovery dialog.\n');
            failCount = failCount + 1;

            % Try to find the WAV by name in DeepSqueak/Audio/
            [~, stem, ext] = fileparts(fname);
            dsAudioPath = fullfile(exampleDir, [stem ext]);
            if isfile(dsAudioPath)
                fprintf('  [HINT] Found in DeepSqueak/Audio/: %s\n', dsAudioPath);
            else
                fprintf('  [HINT] WAV not in DeepSqueak/Audio/ either.\n');
                fprintf('         Copy WAV to DeepSqueak/Audio/ or fix path.\n');
            end
        end
    else
        fprintf('  [FAIL] audiodata has no Filename field!\n');
        failCount = failCount + 1;
    end

    % Check 6: audiodata type comparison with example
    if exist('exData', 'var') && isfield(exData, 'audiodata')
        ourFields = sort(fieldnames(data.audiodata));
        exFields = sort(fieldnames(exData.audiodata));
        if isequal(ourFields, exFields)
            fprintf('  [PASS] audiodata fields match example\n');
            passCount = passCount + 1;
        else
            fprintf('  [WARN] audiodata fields differ from example\n');
            fprintf('         Ours:    ');
            fprintf('%s ', ourFields{:});
            fprintf('\n');
            fprintf('         Example: ');
            fprintf('%s ', exFields{:});
            fprintf('\n');
            warnCount = warnCount + 1;
        end
    end

    fprintf('\n');
end

%% === PART 3: Try loadCallfile directly ===
fprintf('--- PART 3: Direct loadCallfile Test ---\n\n');

if ~isempty(ourFiles)
    testMat = fullfile(ourDir, ourFiles(1).name);
    fprintf('Testing loadCallfile on: %s\n', ourFiles(1).name);

    try
        % loadCallfile(matPath, handles, force) — pass empty handles
        [Ession, audiodata_out] = loadCallfile(testMat, [], true);
        fprintf('[PASS] loadCallfile returned %d calls\n', height(Ession));
        passCount = passCount + 1;
    catch ME
        fprintf('[FAIL] loadCallfile threw error:\n');
        fprintf('       %s\n', ME.message);
        fprintf('       This is likely what DeepSqueak GUI encounters.\n');
        failCount = failCount + 1;
    end
else
    fprintf('[SKIP] No files to test.\n');
end

%% === Summary ===
fprintf('\n======================================\n');
fprintf(' Summary: %d PASS | %d FAIL | %d WARN\n', passCount, failCount, warnCount);
fprintf('======================================\n');

if failCount == 0
    fprintf('\nAll checks passed! If DeepSqueak still shows 0/0, the issue\n');
    fprintf('may be in the GUI display logic (handles/figure state).\n');
else
    fprintf('\nFix the FAIL items above, then re-run this diagnostic.\n');
end
fprintf('\n');
