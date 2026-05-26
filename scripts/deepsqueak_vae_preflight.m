function deepsqueak_vae_preflight(dsFolder, matDir)
% deepsqueak_vae_preflight  Phase 0 environment check before running
% deepsqueak_train_vae. Verifies MATLAB version, Deep Learning Toolbox,
% GPU availability, DS toolbox layout, and detection .mat reachability.
%
%   deepsqueak_vae_preflight(dsFolder, matDir)
%
%   Prints a PASS/FAIL line per check. Exit code is 0 if everything
%   passes, non-zero if any FAIL occurs. Safe to run repeatedly.
%
%   Arguments:
%     dsFolder  - path to DeepSqueak install (with Functions/)
%     matDir    - path to folder containing .mat detection files
%
%   Usage:
%     >> deepsqueak_vae_preflight('C:\path\to\DeepSqueak', ...
%                                 '\\wsl.localhost\Ubuntu\home\shachar\projects\mickey_london_lab\deepsqueak_mats_combined')
%
%     Or:
%       matlab -batch "deepsqueak_vae_preflight('dsDir','matDir')"

    nFail = 0;
    fprintf('\n=== DS VAE Phase 0 Preflight ===\n\n');

    %% 1. MATLAB version
    v = version('-release');
    yearStr = regexp(v, '\d{4}', 'match', 'once');
    yearNum = str2double(yearStr);
    if ~isempty(yearNum) && yearNum >= 2020
        fprintf('  [PASS] MATLAB version: %s (R2020a+ recommended)\n', v);
    else
        fprintf('  [WARN] MATLAB version: %s (may lack dlnetwork features needed)\n', v);
    end

    %% 2. Deep Learning Toolbox
    if ~isempty(ver('nnet'))
        fprintf('  [PASS] Deep Learning Toolbox installed.\n');
    else
        fprintf('  [FAIL] Deep Learning Toolbox NOT installed (required for VAE).\n');
        nFail = nFail + 1;
    end

    %% 3. GPU availability
    try
        if canUseGPU
            gpu = gpuDevice();
            fprintf('  [PASS] GPU: %s, %.1f GB available, CUDA %s\n', ...
                gpu.Name, gpu.AvailableMemory / 1e9, gpu.ComputeCapability);
        else
            fprintf('  [FAIL] No GPU detected (CPU training is impractical).\n');
            nFail = nFail + 1;
        end
    catch ME
        fprintf('  [FAIL] GPU check error: %s\n', ME.message);
        nFail = nFail + 1;
    end

    %% 4. DS toolbox layout
    if ~isfolder(dsFolder)
        fprintf('  [FAIL] dsFolder does not exist: %s\n', dsFolder);
        nFail = nFail + 1;
    else
        fprintf('  [PASS] dsFolder exists: %s\n', dsFolder);
        % Required subpaths
        required = { ...
            'Functions', ...
            'Functions/Variational Autoencoder', ...
            'Functions/Variational Autoencoder/VAE_model.m', ...
            'Functions/Variational Autoencoder/train_vae.m', ...
            'Functions/Variational Autoencoder/extract_VAE_embeddings.m', ...
            'Functions/Variational Autoencoder/sampling.m', ...
            'Functions/Variational Autoencoder/ELBOloss.m', ...
            'Functions/Call Classification/CreateClusteringData.m', ...
            'Functions/Network Training/BlankNet.mat'};
        for k = 1:length(required)
            full = fullfile(dsFolder, required{k});
            if exist(full, 'file') == 2 || exist(full, 'dir') == 7
                fprintf('  [PASS]   %s\n', required{k});
            else
                fprintf('  [FAIL]   missing: %s\n', required{k});
                nFail = nFail + 1;
            end
        end
    end

    %% 5. Detection .mat files
    if ~isfolder(matDir)
        fprintf('  [FAIL] matDir does not exist: %s\n', matDir);
        nFail = nFail + 1;
    else
        matFiles = dir(fullfile(matDir, '*.mat'));
        matFiles = matFiles(~startsWith({matFiles.name}, 'Example '));
        if isempty(matFiles)
            fprintf('  [FAIL] No project .mat files in: %s\n', matDir);
            nFail = nFail + 1;
        else
            fprintf('  [PASS] Found %d .mat files in matDir.\n', length(matFiles));

            % Check first file: audio path reachable?
            try
                samplePath = fullfile(matDir, matFiles(1).name);
                [Calls, audiodata] = loadCallfile(samplePath, []);
                if isempty(Calls) || height(Calls) == 0
                    fprintf('  [WARN] Sample .mat has 0 calls: %s\n', matFiles(1).name);
                elseif ~isfile(audiodata.Filename)
                    fprintf('  [FAIL] Audio file referenced by sample .mat not reachable:\n         %s\n', audiodata.Filename);
                    nFail = nFail + 1;
                else
                    fprintf('  [PASS] Audio path reachable for sample: %s (%d calls)\n', ...
                        matFiles(1).name, height(Calls));
                end
            catch ME
                fprintf('  [WARN] Could not test sample audio path: %s\n', ME.message);
            end

            % Detect cohort labels in filenames
            cohorts = unique(regexp({matFiles.name}, '(5970|lab_131204|3452|9252)', 'match', 'once'));
            cohorts = cohorts(~cellfun(@isempty, cohorts));
            if isempty(cohorts)
                fprintf('  [WARN] No recognizable cohort labels in .mat filenames.\n');
                fprintf('         Pass opts.cohortFromFilename to deepsqueak_train_vae.\n');
            else
                fprintf('  [PASS] Cohorts detected in filenames: %s\n', strjoin(cohorts, ', '));
            end
        end
    end

    %% 6. Disk space at output candidate (tempdir as proxy)
    try
        fb = java.io.File(tempdir);
        freeGb = double(fb.getUsableSpace()) / 1e9;
        if freeGb >= 5
            fprintf('  [PASS] Free space at tempdir: %.1f GB\n', freeGb);
        else
            fprintf('  [WARN] Free space at tempdir: %.1f GB (recommend >=5 GB)\n', freeGb);
        end
    catch
        fprintf('  [SKIP] Could not check tempdir free space.\n');
    end

    %% Summary
    fprintf('\n=== Preflight summary ===\n');
    if nFail == 0
        fprintf('  ALL CHECKS PASSED. Ready to run deepsqueak_train_vae.\n\n');
    else
        fprintf('  %d FAIL(s). Resolve before running deepsqueak_train_vae.\n\n', nFail);
    end
end
