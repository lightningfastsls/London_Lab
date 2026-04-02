function deepsqueak_export_stats(matFiles, handles, outputExcel)
% deepsqueak_export_stats  Headless Excel export of DeepSqueak call statistics.
%
%   deepsqueak_export_stats(matFiles, handles, outputExcel)
%
%   Computes 18 acoustic features per call using DeepSqueak's CalculateStats
%   and writes the standard DeepSqueak Excel format. Designed for headless
%   batch use — no GUI dialogs.
%
%   Arguments:
%     matFiles    - cell array of full paths to .mat detection files
%     handles     - struct with fields:
%                     handles.data.settings.EntropyThreshold  (default: 0.215)
%                     handles.data.settings.AmplitudeThreshold (default: 0.825)
%     outputExcel - full path to output .xlsx file
%
%   Output columns (18, matching DeepSqueak's excel_Callback):
%     File, ID, Label, Accepted, Score,
%     Begin Time (s), End Time (s), Call Length (s),
%     Principal Frequency (kHz), Low Freq (kHz), High Freq (kHz),
%     Delta Freq (kHz), Frequency Standard Deviation (kHz),
%     Slope (kHz/s), Sinuosity, Mean Power (dB/Hz), Tonality,
%     Peak Freq (kHz)
%
%   Part of the headless DeepSqueak bridge pipeline:
%     1. create_deepsqueak_mats.m     (Raven TSV -> .mat)
%     2. deepsqueak_batch_classify.m  (classify calls)
%     3. deepsqueak_export_stats.m    (export Excel)  <-- this file
%     4. import_deepsqueak_results.py (merge with Python detections)
%
%   Based on DeepSqueak v3.1 (commit 1be0267) excel_Callback.m source analysis.
%
%   Usage:
%     >> deepsqueak_export_stats(matFiles, handles, 'output_Stats.xlsx')

    fprintf('\n=== DeepSqueak Headless Export ===\n\n');

    % Validate inputs
    if isempty(matFiles)
        error('deepsqueak_export_stats:noFiles', 'matFiles is empty.');
    end
    if ~isfield(handles, 'data') || ~isfield(handles.data, 'settings')
        error('deepsqueak_export_stats:badHandles', ...
            'handles must contain handles.data.settings with EntropyThreshold and AmplitudeThreshold.');
    end

    EntropyThreshold = handles.data.settings.EntropyThreshold;
    AmplitudeThreshold = handles.data.settings.AmplitudeThreshold;

    % Column headers matching DeepSqueak's standard Excel output
    colHeaders = {'File', 'ID', 'Label', 'Accepted', 'Score', ...
        'Begin Time (s)', 'End Time (s)', 'Call Length (s)', ...
        'Principal Frequency (kHz)', 'Low Freq (kHz)', 'High Freq (kHz)', ...
        'Delta Freq (kHz)', 'Frequency Standard Deviation (kHz)', ...
        'Slope (kHz/s)', 'Sinuosity', 'Mean Power (dB/Hz)', 'Tonality', ...
        'Peak Freq (kHz)'};

    % Accumulate all rows
    allRows = {};
    totalCalls = 0;
    totalSkipped = 0;

    for j = 1:length(matFiles)
        matPath = matFiles{j};
        [~, matName, ~] = fileparts(matPath);
        fprintf('  [%d/%d] %s ... ', j, length(matFiles), matName);

        % Load calls and audio metadata
        [Calls, audiodata_j] = loadCallfile(matPath, []);

        if isempty(Calls) || height(Calls) == 0
            fprintf('0 calls (skipped)\n');
            continue;
        end

        % Create audio reader for spectrogram generation
        audioReader = squeakData([]);
        audioReader.audiodata = audiodata_j;

        fileCallCount = 0;

        for i = 1:height(Calls)
            % Skip zero-dimension boxes (DeepSqueak silently removes these)
            if Calls.Box(i,3) == 0 || Calls.Box(i,4) == 0
                totalSkipped = totalSkipped + 1;
                continue;
            end

            try
                % Generate spectrogram from raw audio
                [I, windowsize, noverlap, nfft, rate, box] = ...
                    CreateFocusSpectrogram(Calls(i,:), handles, true, [], audioReader);

                % Compute all 16 acoustic features
                stats = CalculateStats(I, windowsize, noverlap, nfft, rate, box, ...
                    EntropyThreshold, AmplitudeThreshold);

                % Extract label (handle categorical or string)
                if iscategorical(Calls.Type(i))
                    label = char(Calls.Type(i));
                else
                    label = Calls.Type(i);
                end

                % Build row matching DeepSqueak's 18-column format
                row = {matName, ...               % File
                       i, ...                     % ID
                       label, ...                 % Label
                       Calls.Accept(i), ...       % Accepted
                       Calls.Score(i), ...        % Score
                       stats.BeginTime, ...       % Begin Time (s)
                       stats.EndTime, ...         % End Time (s)
                       stats.DeltaTime, ...       % Call Length (s)
                       stats.PrincipalFreq, ...   % Principal Frequency (kHz)
                       stats.LowFreq, ...         % Low Freq (kHz)
                       stats.HighFreq, ...        % High Freq (kHz)
                       stats.DeltaFreq, ...       % Delta Freq (kHz)
                       stats.stdev, ...           % Frequency Std Dev (kHz)
                       stats.Slope, ...           % Slope (kHz/s)
                       stats.Sinuosity, ...       % Sinuosity
                       stats.MeanPower, ...       % Mean Power (dB/Hz)
                       stats.SignalToNoise, ...   % Tonality
                       stats.PeakFreq};           % Peak Freq (kHz)

                allRows = [allRows; row]; %#ok<AGROW>
                fileCallCount = fileCallCount + 1;
            catch ME
                fprintf('\n    [WARN] Call %d failed: %s\n', i, ME.message);
                totalSkipped = totalSkipped + 1;
            end
        end

        totalCalls = totalCalls + fileCallCount;
        fprintf('%d calls exported\n', fileCallCount);
    end

    % Write Excel file
    if isempty(allRows)
        fprintf('\n[WARN] No calls to export. No file written.\n');
        return;
    end

    % Create output directory if needed
    outDir = fileparts(outputExcel);
    if ~isempty(outDir) && ~isfolder(outDir)
        mkdir(outDir);
        fprintf('  Created output directory: %s\n', outDir);
    end

    % Build table and write
    t = cell2table(allRows, 'VariableNames', colHeaders);
    writetable(t, outputExcel);

    fprintf('\n=== Export complete ===\n');
    fprintf('  Total calls exported: %d\n', totalCalls);
    fprintf('  Skipped (zero-dim or error): %d\n', totalSkipped);
    fprintf('  Output: %s\n\n', outputExcel);
end
