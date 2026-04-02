% test_deepsqueak_batch.m
% Post-run validation for the headless DeepSqueak classification pipeline.
%
% Validates:
%   1. .mat files have cluster labels (Calls.Type updated from 'USV')
%   2. Excel output exists with correct 18-column format
%   3. Excel values are within plausible ranges for mouse USVs
%   4. Clustering model was saved
%   5. Round-trip: Excel timestamps match .mat Box timestamps
%
% Run AFTER deepsqueak_batch_classify.m completes.
%
% Usage:
%   >> test_deepsqueak_batch( ...
%          '\\wsl$\Ubuntu\home\shachar\projects\mickey_london_lab\deepsqueak_mats', ...
%          '\\wsl$\Ubuntu\home\shachar\projects\mickey_london_lab\deepsqueak_output')

function test_deepsqueak_batch(matDir, outputDir)

    fprintf('\n');
    fprintf('==================================================\n');
    fprintf(' DeepSqueak Batch Pipeline Validation\n');
    fprintf('==================================================\n\n');

    passCount = 0;
    failCount = 0;
    warnCount = 0;

    %% === Test 1: .mat files have cluster labels ===
    fprintf('--- Test 1: Cluster labels in .mat files ---\n\n');

    matFiles = dir(fullfile(matDir, '*.mat'));
    if isempty(matFiles)
        fprintf('  [FAIL] No .mat files found in: %s\n', matDir);
        failCount = failCount + 1;
    else
        fprintf('  Found %d .mat files\n', length(matFiles));
        passCount = passCount + 1;

        allLabels = {};
        totalCalls = 0;
        unlabeledCount = 0;

        for i = 1:length(matFiles)
            matPath = fullfile(matDir, matFiles(i).name);
            data = load(matPath, 'Calls');

            if ~isfield(data, 'Calls') || ~istable(data.Calls)
                fprintf('  [FAIL] %s: Calls is not a table\n', matFiles(i).name);
                failCount = failCount + 1;
                continue;
            end

            nCalls = height(data.Calls);
            totalCalls = totalCalls + nCalls;

            for j = 1:nCalls
                if iscategorical(data.Calls.Type(j))
                    lbl = char(data.Calls.Type(j));
                else
                    lbl = data.Calls.Type(j);
                end

                allLabels{end+1} = lbl; %#ok<AGROW>

                if strcmp(lbl, 'USV') || strcmp(lbl, '<undefined>')
                    unlabeledCount = unlabeledCount + 1;
                end
            end
        end

        uniqueLabels = unique(allLabels);
        fprintf('  Total calls: %d\n', totalCalls);
        fprintf('  Unique labels: %d\n', length(uniqueLabels));
        fprintf('  Labels: ');
        for k = 1:min(15, length(uniqueLabels))
            fprintf('%s  ', uniqueLabels{k});
        end
        if length(uniqueLabels) > 15
            fprintf('... (+%d more)', length(uniqueLabels) - 15);
        end
        fprintf('\n');

        if unlabeledCount == totalCalls && totalCalls > 0
            fprintf('  [FAIL] All %d calls still labeled "USV" — classification did not run\n', totalCalls);
            failCount = failCount + 1;
        elseif unlabeledCount > 0
            fprintf('  [WARN] %d/%d calls still labeled "USV" or undefined\n', unlabeledCount, totalCalls);
            warnCount = warnCount + 1;
        else
            fprintf('  [PASS] All calls have cluster labels\n');
            passCount = passCount + 1;
        end
    end
    fprintf('\n');

    %% === Test 2: Excel output exists with correct columns ===
    fprintf('--- Test 2: Excel output format ---\n\n');

    excelPath = fullfile(outputDir, 'classified_Stats.xlsx');
    if ~isfile(excelPath)
        fprintf('  [FAIL] Excel file not found: %s\n', excelPath);
        failCount = failCount + 1;
        fprintf('\n');
    else
        fprintf('  [PASS] Excel file exists: %s\n', excelPath);
        passCount = passCount + 1;

        t = readtable(excelPath);
        fprintf('  Rows: %d, Columns: %d\n', height(t), width(t));

        expectedCols = {'File', 'ID', 'Label', 'Accepted', 'Score', ...
            'BeginTime_s_', 'EndTime_s_', 'CallLength_s_', ...
            'PrincipalFrequency_kHz_', 'LowFreq_kHz_', 'HighFreq_kHz_', ...
            'DeltaFreq_kHz_', 'FrequencyStandardDeviation_kHz_', ...
            'Slope_kHz_s_', 'Sinuosity', 'MeanPower_dB_Hz_', 'Tonality', ...
            'PeakFreq_kHz_'};

        % Check column count
        if width(t) == 18
            fprintf('  [PASS] Correct column count (18)\n');
            passCount = passCount + 1;
        else
            fprintf('  [FAIL] Expected 18 columns, got %d\n', width(t));
            failCount = failCount + 1;
            fprintf('  Actual columns: ');
            fprintf('%s  ', t.Properties.VariableNames{:});
            fprintf('\n');
        end

        % Check row count matches total calls
        if height(t) == totalCalls
            fprintf('  [PASS] Row count matches total calls (%d)\n', totalCalls);
            passCount = passCount + 1;
        elseif height(t) > 0
            fprintf('  [WARN] Row count (%d) differs from total calls (%d) — some may have been skipped\n', ...
                height(t), totalCalls);
            warnCount = warnCount + 1;
        else
            fprintf('  [FAIL] Excel file is empty\n');
            failCount = failCount + 1;
        end
        fprintf('\n');

        %% === Test 3: Plausible value ranges for mouse USVs ===
        fprintf('--- Test 3: Plausible value ranges ---\n\n');

        if height(t) > 0
            % Mouse USVs: 25-125 kHz, 5-200 ms typical duration
            % These are loose bounds to catch obvious errors

            % Find numeric columns by trying common MATLAB-mangled names
            % (MATLAB readtable converts spaces/parens to underscores)
            colNames = t.Properties.VariableNames;

            checks = {
                'BeginTime',      0,    3600,  's';      % 0 to 1 hour
                'EndTime',        0,    3600,  's';
                'CallLength',     0,    2,     's';      % up to 2s (generous)
                'PrincipalFreq',  15,   150,   'kHz';    % mouse USV range (loose)
                'LowFreq',        10,   150,   'kHz';
                'HighFreq',       15,   200,   'kHz';
                'DeltaFreq',      0,    150,   'kHz';
                'Slope',          -Inf, Inf,   'kHz/s';  % any value OK
                'Sinuosity',      0,    Inf,   '';       % >= 0
                'Tonality',       0,    1,     '';       % 0 to 1
            };

            for c = 1:size(checks, 1)
                pattern = checks{c,1};
                lo = checks{c,2};
                hi = checks{c,3};
                unit = checks{c,4};

                % Find column matching pattern (MATLAB mangles names)
                matchIdx = find(cellfun(@(x) contains(x, pattern, 'IgnoreCase', true), colNames), 1);
                if isempty(matchIdx)
                    fprintf('  [WARN] Column matching "%s" not found\n', pattern);
                    warnCount = warnCount + 1;
                    continue;
                end

                vals = t{:, matchIdx};
                if ~isnumeric(vals)
                    fprintf('  [WARN] %s is not numeric\n', colNames{matchIdx});
                    warnCount = warnCount + 1;
                    continue;
                end

                vals_clean = vals(~isnan(vals) & ~isinf(vals));
                if isempty(vals_clean)
                    fprintf('  [WARN] %s: all NaN/Inf\n', colNames{matchIdx});
                    warnCount = warnCount + 1;
                    continue;
                end

                minVal = min(vals_clean);
                maxVal = max(vals_clean);
                nanCount = sum(isnan(vals));

                if minVal >= lo && maxVal <= hi
                    fprintf('  [PASS] %s: [%.2f, %.2f] %s', ...
                        colNames{matchIdx}, minVal, maxVal, unit);
                    if nanCount > 0
                        fprintf(' (%d NaN)', nanCount);
                    end
                    fprintf('\n');
                    passCount = passCount + 1;
                else
                    fprintf('  [FAIL] %s: [%.2f, %.2f] %s — outside expected [%.1f, %.1f]\n', ...
                        colNames{matchIdx}, minVal, maxVal, unit, lo, hi);
                    failCount = failCount + 1;
                end
            end
        end
        fprintf('\n');
    end

    %% === Test 4: Clustering model saved ===
    fprintf('--- Test 4: Clustering model ---\n\n');

    kmeansModel = fullfile(outputDir, 'clustering_model_kmeans.mat');
    artwarpModel = fullfile(outputDir, 'clustering_model_artwarp.mat');

    if isfile(kmeansModel)
        m = load(kmeansModel);
        fprintf('  [PASS] K-means model found: k=%d\n', m.optimalK);
        fprintf('         Weights: freq=%.0f, slope=%.0f, duration=%.0f\n', ...
            m.FREQ_WEIGHT, m.SLOPE_WEIGHT, m.DURATION_WEIGHT);
        passCount = passCount + 1;
    elseif isfile(artwarpModel)
        m = load(artwarpModel);
        fprintf('  [PASS] ARTwarp model found: %d clusters\n', m.nClusters);
        passCount = passCount + 1;
    else
        fprintf('  [FAIL] No clustering model found in: %s\n', outputDir);
        failCount = failCount + 1;
    end
    fprintf('\n');

    %% === Test 5: Round-trip timestamp consistency ===
    fprintf('--- Test 5: Timestamp round-trip (.mat vs Excel) ---\n\n');

    if isfile(excelPath) && ~isempty(matFiles)
        t = readtable(excelPath);

        % Find Begin Time column
        beginCol = find(cellfun(@(x) contains(x, 'BeginTime', 'IgnoreCase', true), ...
            t.Properties.VariableNames), 1);

        if ~isempty(beginCol) && isnumeric(t{:, beginCol})
            % Compare first .mat file's Box start times with Excel
            firstMat = fullfile(matDir, matFiles(1).name);
            data = load(firstMat, 'Calls');
            [~, matStem, ~] = fileparts(matFiles(1).name);

            % Find Excel rows for this file
            fileCol = find(cellfun(@(x) contains(x, 'File', 'IgnoreCase', true), ...
                t.Properties.VariableNames), 1);

            if ~isempty(fileCol)
                if iscellstr(t{:, fileCol}) || isstring(t{:, fileCol})
                    mask = strcmp(t{:, fileCol}, matStem);
                else
                    mask = false(height(t), 1);
                end

                excelTimes = sort(t{mask, beginCol});
                matTimes = sort(data.Calls.Box(:, 1));

                if length(excelTimes) == length(matTimes)
                    maxDiff = max(abs(excelTimes - matTimes));
                    if maxDiff < 0.001  % sub-millisecond agreement
                        fprintf('  [PASS] Timestamps match (max diff: %.6f s)\n', maxDiff);
                        passCount = passCount + 1;
                    else
                        fprintf('  [WARN] Timestamps differ by up to %.4f s\n', maxDiff);
                        warnCount = warnCount + 1;
                    end
                else
                    fprintf('  [WARN] Count mismatch: %d in .mat vs %d in Excel for %s\n', ...
                        length(matTimes), length(excelTimes), matStem);
                    warnCount = warnCount + 1;
                end
            else
                fprintf('  [WARN] Could not find File column in Excel\n');
                warnCount = warnCount + 1;
            end
        else
            fprintf('  [WARN] Could not find numeric Begin Time column\n');
            warnCount = warnCount + 1;
        end
    else
        fprintf('  [SKIP] Cannot run without Excel output and .mat files\n');
    end
    fprintf('\n');

    %% === Summary ===
    fprintf('==================================================\n');
    fprintf(' Validation: %d PASS | %d FAIL | %d WARN\n', passCount, failCount, warnCount);
    fprintf('==================================================\n');

    if failCount == 0
        fprintf('\n  All critical checks passed.\n');
        if warnCount > 0
            fprintf('  Review warnings above — they may indicate minor issues.\n');
        end
        fprintf('\n  Ready for Python import:\n');
        fprintf('    .venv/bin/python scripts/import_deepsqueak_results.py \\\n');
        fprintf('        --results-dir %s \\\n', strrep(outputDir, '\', '/'));
        fprintf('        --detections-dir USV_Detections \\\n');
        fprintf('        --output classified_detections.csv\n');
    else
        fprintf('\n  %d failures detected. Fix before proceeding to Python import.\n', failCount);
    end
    fprintf('\n');
end
