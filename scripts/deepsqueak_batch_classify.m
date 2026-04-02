function deepsqueak_batch_classify(matDir, dsFolder, outputDir, method)
% deepsqueak_batch_classify  Headless batch classification of USV detections.
%
%   deepsqueak_batch_classify(matDir, dsFolder, outputDir, method)
%
%   Loads .mat detection files (created by create_deepsqueak_mats.m), runs
%   DeepSqueak's classification algorithms WITHOUT the GUI, saves cluster
%   labels back to the .mat files, and exports results to Excel.
%
%   Arguments:
%     matDir    - folder containing .mat detection files
%     dsFolder  - path to DeepSqueak installation (for function access)
%     outputDir - output directory for Excel files and clustering model
%     method    - 'kmeans' (default) or 'artwarp'
%
%   Clustering methods:
%     'kmeans'  - Contour parameter k-means with automatic k selection
%                 (elbow method via kmeans_opt). Default feature weights:
%                 shape=3, freq=2, duration=1. This is DeepSqueak's standard
%                 unsupervised clustering.
%
%     'artwarp' - Adaptive Resonance Theory with Dynamic Time Warping.
%                 Self-organizing: determines number of clusters dynamically.
%                 Better for non-spherical cluster shapes but slower.
%
%   Output:
%     - Cluster labels saved back to each .mat file (Calls.Type updated)
%     - Excel file in outputDir with 18-column DeepSqueak stats format
%     - Clustering model saved to outputDir for reuse
%
%   Part of the headless DeepSqueak bridge pipeline:
%     1. create_deepsqueak_mats.m     (Raven TSV -> .mat)
%     2. deepsqueak_batch_classify.m  (classify calls)  <-- this file
%     3. deepsqueak_export_stats.m    (export Excel)
%     4. import_deepsqueak_results.py (merge with Python detections)
%
%   Based on DeepSqueak v3.1 (commit 1be0267) source code analysis.
%   Calls only verified headless-safe functions.
%
%   Usage:
%     >> deepsqueak_batch_classify( ...
%            '\\wsl.localhost\Ubuntu\home\shachar\projects\mickey_london_lab\deepsqueak_mats', ...
%            'C:\path\to\DeepSqueak', ...
%            '\\wsl.localhost\Ubuntu\home\shachar\projects\mickey_london_lab\deepsqueak_output', ...
%            'kmeans')
%
%     Or from command line:
%       matlab -batch "deepsqueak_batch_classify('matDir','dsDir','outDir','kmeans')"

    %% === Configuration ===

    % Default method
    if nargin < 4 || isempty(method)
        method = 'kmeans';
    end

    % K-means feature weights (DeepSqueak defaults)
    SLOPE_WEIGHT = 3;
    FREQ_WEIGHT = 2;
    DURATION_WEIGHT = 1;

    % ARTwarp default settings
    ARTWARP_SETTINGS = {5, 2.5, 8, 0.001, 5, 4, 1, 1};
    % {MatchThresh, CombineVigilance, OutlierThresh, LearningRate,
    %  MaxIterations, ShapeImportance, FreqImportance, TimeImportance}

    % CalculateStats thresholds (DeepSqueak defaults from squeakData)
    ENTROPY_THRESHOLD = 0.215;
    AMPLITUDE_THRESHOLD = 0.825;

    %% === Setup ===

    fprintf('\n');
    fprintf('==================================================\n');
    fprintf(' DeepSqueak Headless Batch Classification\n');
    fprintf('==================================================\n');
    fprintf('  Method:    %s\n', method);
    fprintf('  Mat dir:   %s\n', matDir);
    fprintf('  DS folder: %s\n', dsFolder);
    fprintf('  Output:    %s\n', outputDir);
    fprintf('==================================================\n\n');

    % Validate inputs
    if ~isfolder(matDir)
        error('deepsqueak_batch_classify:badMatDir', ...
            'matDir does not exist: %s', matDir);
    end
    if ~isfolder(dsFolder)
        error('deepsqueak_batch_classify:badDsFolder', ...
            'dsFolder does not exist: %s', dsFolder);
    end
    if ~ismember(method, {'kmeans', 'artwarp'})
        error('deepsqueak_batch_classify:badMethod', ...
            'method must be ''kmeans'' or ''artwarp'', got: %s', method);
    end

    % Create output directory
    if ~isfolder(outputDir)
        mkdir(outputDir);
        fprintf('Created output directory: %s\n\n', outputDir);
    end

    % Add DeepSqueak functions to MATLAB path
    dsFunc = fullfile(dsFolder, 'Functions');
    if ~isfolder(dsFunc)
        error('deepsqueak_batch_classify:noDsFunctions', ...
            'DeepSqueak Functions/ folder not found at: %s', dsFunc);
    end
    addpath(genpath(dsFunc));
    fprintf('Added DeepSqueak functions to path.\n');

    % Suppress figure display in batch mode
    set(0, 'DefaultFigureVisible', 'off');

    % Build minimal mock handles (3 fields needed by CreateFocusSpectrogram
    % and CalculateStats, verified from source analysis)
    handles.data.settings.detectionfolder = matDir;
    handles.data.settings.EntropyThreshold = ENTROPY_THRESHOLD;
    handles.data.settings.AmplitudeThreshold = AMPLITUDE_THRESHOLD;
    handles.data.squeakfolder = dsFolder;

    % Find .mat files (skip bundled DeepSqueak examples)
    matFiles_struct = dir(fullfile(matDir, '*.mat'));
    if isempty(matFiles_struct)
        error('deepsqueak_batch_classify:noMatFiles', ...
            'No .mat files found in: %s', matDir);
    end

    matFiles = {};
    skipped = {};
    for k = 1:length(matFiles_struct)
        fname = matFiles_struct(k).name;
        if startsWith(fname, 'Example ', 'IgnoreCase', true)
            skipped{end+1} = fname; %#ok<AGROW>
        else
            matFiles{end+1} = fullfile(matDir, fname); %#ok<AGROW>
        end
    end
    matFiles = matFiles(:);  % column cell array
    if ~isempty(skipped)
        fprintf('Skipped %d non-project files (Example recordings).\n', length(skipped));
    end
    if isempty(matFiles)
        error('deepsqueak_batch_classify:noMatFiles', ...
            'No project .mat files found in: %s', matDir);
    end
    fprintf('Found %d .mat detection files.\n\n', length(matFiles));

    %% === Phase 1: Build ClusteringData ===
    % Replicates CreateClusteringData logic but without uigetfile dialogs.
    % For each call: load audio, generate spectrogram, extract features.

    fprintf('--- Phase 1: Building ClusteringData ---\n\n');

    ClusteringData_cells = {};
    clustAssign = [];

    for j = 1:length(matFiles)
        matPath = matFiles{j};
        [~, matName, ~] = fileparts(matPath);
        fprintf('  [%d/%d] %s ... ', j, length(matFiles), matName);

        % Load detection file (handles=[] skips audio recovery dialog)
        [Calls, audiodata_j] = loadCallfile(matPath, []);

        if isempty(Calls) || height(Calls) == 0
            fprintf('0 calls (skipped)\n');
            continue;
        end

        % Verify audio file is accessible
        if ~isfile(audiodata_j.Filename)
            fprintf('\n    [ERROR] Audio file not found: %s\n', audiodata_j.Filename);
            fprintf('    Skipping this file. Fix audiodata.Filename path.\n');
            continue;
        end

        % Create audio reader (DeepSqueak's on-demand audio access)
        audioReader = squeakData([]);
        audioReader.audiodata = audiodata_j;

        fileCallCount = 0;

        for i = 1:height(Calls)
            % Skip zero-dimension boxes
            if Calls.Box(i,3) == 0 || Calls.Box(i,4) == 0
                continue;
            end

            try
                % Generate spectrogram from raw audio
                % make_spectrogram=true means handles is NOT accessed
                [I, wind, noverlap, nfft, rate, box, s, fr, ti, ~, pow] = ...
                    CreateFocusSpectrogram(Calls(i,:), handles, true, [], audioReader);

                % Image processing (matches CreateClusteringData lines 109-115)
                pow(pow == 0) = .01;
                pow = log10(pow);
                pow = rescale(imcomplement(abs(pow)));
                pow = flipud(pow);
                im = imadjust(pow, [.5 .9]);

                % Compute acoustic features (fully standalone, no handles)
                stats = CalculateStats(I, wind, noverlap, nfft, rate, box, ...
                    ENTROPY_THRESHOLD, AMPLITUDE_THRESHOLD);

                % Compute frequency contour in kHz (for clustering features)
                spectrange = audioReader.audiodata.SampleRate / 2000;
                FreqScale = spectrange / (1 + floor(nfft / 2));
                TimeScale = (wind - noverlap) / audioReader.audiodata.SampleRate;
                xFreq = FreqScale * stats.ridgeFreq_smooth + Calls.Box(i,2);
                xTime = stats.ridgeTime * TimeScale;

                % Accumulate into ClusteringData
                % Column order matches DeepSqueak's ClusteringData table
                ClusteringData_cells = [ClusteringData_cells; ...
                    {uint8(im .* 256)}, ...  % Spectrogram
                    {box}, ...               % Box [start_s, lowFreq_kHz, dur_s, bw_kHz]
                    {box(2)}, ...            % MinFreq (kHz)
                    {stats.DeltaTime}, ...   % Duration (s)
                    {xFreq}, ...             % xFreq (frequency contour)
                    {xTime}, ...             % xTime (time contour)
                    {matPath}, ...           % Filename (.mat path)
                    {i}, ...                 % callID (row index in Calls)
                    {stats.Power}, ...       % Power
                    {box(4)}]; %#ok<AGROW>   % Bandwidth (kHz)

                clustAssign = [clustAssign; Calls.Type(i)]; %#ok<AGROW>
                fileCallCount = fileCallCount + 1;

            catch ME
                fprintf('\n    [WARN] Call %d failed: %s\n', i, ME.message);
            end
        end

        fprintf('%d calls processed\n', fileCallCount);
    end

    % Convert to table
    if isempty(ClusteringData_cells)
        error('deepsqueak_batch_classify:noCalls', ...
            'No valid calls found across all .mat files.');
    end

    ClusteringData = cell2table(ClusteringData_cells, ...
        'VariableNames', {'Spectrogram', 'Box', 'MinFreq', 'Duration', ...
        'xFreq', 'xTime', 'Filename', 'callID', 'Power', 'Bandwidth'});

    totalCalls = height(ClusteringData);
    fprintf('\n  Total calls for clustering: %d\n\n', totalCalls);

    %% === Phase 2: Run Clustering ===

    fprintf('--- Phase 2: Clustering (%s) ---\n\n', method);

    switch method
        case 'kmeans'
            % Replicate get_kmeans_data + kmeans_opt from
            % UnsupervisedClustering_Callback (lines 90-110)

            % Resample frequency contours to 13 points (for 12 slope values)
            ReshapedX = cell2mat(cellfun(@(x) imresize(x', [1 13]), ...
                ClusteringData.xFreq, 'UniformOutput', false));

            % Three feature types, z-scored and weighted
            slope = zscore(diff(ReshapedX, 1, 2));               % 12 columns
            freq = zscore(cell2mat(cellfun(@(x) imresize(x', [1 12]), ...
                ClusteringData.xFreq, 'UniformOutput', false)));  % 12 columns
            duration = zscore(repmat(ClusteringData.Duration, [1 12])); % 12 columns

            data = [freq .* FREQ_WEIGHT, ...
                    slope .* SLOPE_WEIGHT, ...
                    duration .* DURATION_WEIGHT];

            % Handle NaN values (from calls with poor contour extraction)
            nanRows = any(isnan(data), 2);
            if sum(nanRows) > 0
                fprintf('  [WARN] Removing %d calls with NaN features.\n', sum(nanRows));
                data(nanRows, :) = [];
                validIdx = find(~nanRows);
            else
                validIdx = (1:totalCalls)';
            end

            % Optimal k-means (elbow method)
            maxK = min(100, size(data, 1));
            fprintf('  Running kmeans_opt (maxK=%d) ...\n', maxK);
            [~, C] = kmeans_opt(data, maxK, 0, 3);
            optimalK = size(C, 1);
            fprintf('  Optimal k = %d\n', optimalK);

            % Assign all points to nearest centroid
            [clustIdx, ~] = knnsearch(C, data, 'Distance', 'euclidean');

            % Build full assignment vector (NaN rows get 'Unclassified')
            clustAssign_new = categorical(repmat({'Unclassified'}, totalCalls, 1));
            for ci = 1:length(validIdx)
                clustAssign_new(validIdx(ci)) = categorical({sprintf('Cluster_%d', clustIdx(ci))});
            end

            % Generate cluster names
            clusterNames = unique(clustAssign_new);

            % Save clustering model for reuse
            % (use temp file + copyfile to work around MATLAB save() UNC limitation)
            modelPath = fullfile(outputDir, 'clustering_model_kmeans.mat');
            tmpModel = fullfile(tempdir, 'clustering_model_kmeans.mat');
            save(tmpModel, 'C', 'FREQ_WEIGHT', 'SLOPE_WEIGHT', 'DURATION_WEIGHT', ...
                'optimalK', '-v7.3');
            copyfile(tmpModel, modelPath);
            delete(tmpModel);
            fprintf('  Clustering model saved: %s\n', modelPath);

        case 'artwarp'
            % Adaptive Resonance Theory with Dynamic Time Warping
            % Calls ARTwarp2 directly (fully headless)

            fprintf('  Running ARTwarp2 ...\n');
            fprintf('  Settings: MatchThresh=%.1f, CombineVigilance=%.1f\n', ...
                ARTWARP_SETTINGS{1}, ARTWARP_SETTINGS{2});

            [net, clustIdx] = ARTwarp2(ClusteringData.xFreq, ARTWARP_SETTINGS);

            nClusters = max(clustIdx);
            fprintf('  Found %d clusters\n', nClusters);

            % Convert to categorical labels
            clustAssign_new = categorical(arrayfun(@(x) sprintf('Cluster_%d', x), ...
                clustIdx, 'UniformOutput', false));
            clusterNames = unique(clustAssign_new);

            % Save clustering model for reuse
            % (use temp file + copyfile to work around MATLAB save() UNC limitation)
            modelPath = fullfile(outputDir, 'clustering_model_artwarp.mat');
            tmpModel = fullfile(tempdir, 'clustering_model_artwarp.mat');
            save(tmpModel, 'net', 'ARTWARP_SETTINGS', 'nClusters', '-v7.3');
            copyfile(tmpModel, modelPath);
            delete(tmpModel);
            fprintf('  Clustering model saved: %s\n', modelPath);

        otherwise
            error('deepsqueak_batch_classify:badMethod', ...
                'Unknown method: %s', method);
    end

    fprintf('\n');

    %% === Phase 3: Save Labels to .mat Files ===

    fprintf('--- Phase 3: Saving cluster labels to .mat files ---\n\n');

    % UpdateCluster reads ClusteringData.Filename and callID to locate
    % which call in which file to update. Fully headless.
    rejected = zeros(1, totalCalls);  % Accept all clusters
    UpdateCluster(ClusteringData, clustAssign_new, clusterNames, rejected);

    fprintf('  Labels saved to %d .mat files.\n\n', length(matFiles));

    %% === Phase 4: Export Excel ===

    fprintf('--- Phase 4: Exporting statistics to Excel ---\n\n');

    outputExcel = fullfile(outputDir, 'classified_Stats.xlsx');
    deepsqueak_export_stats(matFiles, handles, outputExcel);

    %% === Summary ===

    fprintf('==================================================\n');
    fprintf(' Classification Complete\n');
    fprintf('==================================================\n');
    fprintf('  Method:          %s\n', method);
    fprintf('  Total calls:     %d\n', totalCalls);
    fprintf('  Clusters found:  %d\n', length(clusterNames));
    fprintf('  Cluster names:   ');
    for ci = 1:min(10, length(clusterNames))
        fprintf('%s  ', char(clusterNames(ci)));
    end
    if length(clusterNames) > 10
        fprintf('... (+%d more)', length(clusterNames) - 10);
    end
    fprintf('\n');
    fprintf('  Excel output:    %s\n', outputExcel);
    fprintf('  Model saved:     %s\n', modelPath);
    fprintf('==================================================\n');
    fprintf('\n  Next step (Python/WSL):\n');
    fprintf('    .venv/bin/python scripts/import_deepsqueak_results.py \\\n');
    fprintf('        --results-dir %s \\\n', strrep(outputDir, '\', '/'));
    fprintf('        --detections-dir USV_Detections \\\n');
    fprintf('        --output classified_detections.csv\n\n');

    % Restore figure visibility
    set(0, 'DefaultFigureVisible', 'on');
end
