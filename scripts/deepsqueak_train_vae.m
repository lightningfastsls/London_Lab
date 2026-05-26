function deepsqueak_train_vae(matDir, dsFolder, outputDir, opts)
% deepsqueak_train_vae  Headless training of DS's Variational Autoencoder
% across multiple cohorts (wild 5970, lab_131204, etc.) for cross-cohort
% repertoire comparison.
%
%   deepsqueak_train_vae(matDir, dsFolder, outputDir, opts)
%
%   Trains the 32-dim VAE from DeepSqueak v3.1 (commit 1be0267):
%     Functions/Variational Autoencoder/VAE_model.m
%     Functions/Variational Autoencoder/train_vae.m
%     Functions/Variational Autoencoder/extract_VAE_embeddings.m
%
%   Each call in matDir/*.mat gets a 128x128 spectrogram via DS's
%   CreateFocusSpectrogram + standard intensity preprocessing (same as the
%   existing deepsqueak_batch_classify.m wrapper). Cohort is parsed from
%   the .mat filename. Optional frequency-domain notch (rectangular band
%   zeroing) is applied to each image before training to suppress cage
%   tonal artifacts (e.g. lab 50-52 kHz and 63-64.5 kHz lines). Training
%   uses a balanced subsample per cohort; embeddings are then extracted for
%   ALL calls, not just the training subsample.
%
%   Arguments:
%     matDir    - folder containing .mat detection files (all cohorts)
%     dsFolder  - path to DeepSqueak install with Functions/ subfolder
%     outputDir - destination for trained networks + embeddings CSV
%     opts (optional struct, all fields optional):
%       .cohortFromFilename - regex picking the cohort label from .mat
%                             filename. Default '(5970|lab_131204|3452|9252)'
%       .subsamplePerCohort - max calls per cohort in training pool
%                             (default 8000). All cohorts subsampled to
%                             min(N_cohort, this value).
%       .frequencyMaskKhz   - Nx2 matrix [low_khz, high_khz] of bands to
%                             zero in each 128x128 image before training.
%                             Applied UNIFORMLY across all cohorts unless
%                             frequencyMaskKhzByCohort overrides it for a
%                             specific cohort. Default [] (no global mask).
%       .frequencyMaskKhzByCohort - struct keyed by cohort label; each value
%                             is an Nx2 [low_khz, high_khz] band matrix
%                             applied ONLY to that cohort's images. Empty
%                             value or missing field means "fall back to
%                             frequencyMaskKhz". Use this to mask cage
%                             tonal lines per recording chamber. Example:
%                               opts.frequencyMaskKhzByCohort = struct( ...
%                                 'lab_131204', [50.4 51.0], ...
%                                 'x5970',      []);
%                             (Field names must be valid MATLAB identifiers;
%                             numeric-leading cohorts like '5970' become
%                             'x5970' via matlab.lang.makeValidName.)
%       .cohorts            - cell array of cohort labels to include in
%                             training. Files whose cohort does not appear
%                             in this list are skipped at the per-file
%                             loop, saving I/O. Default {} (include all).
%       .numEpochs          - default 250 (DS default)
%       .latentDim          - default 32 (DS default)
%       .miniBatchSize      - default 128 (DS default)
%       .learningRate       - default 2.5e-4 (DS default)
%       .randomSeed         - default 42 (subsample reproducibility)
%       .freqRangeKhz       - 1x2 [low high] kHz range covered by each
%                             spectrogram. Default [15 75] (DS default
%                             when fixed_frequency=true)
%
%   Outputs to outputDir/:
%     vae_encoder.mat            - trained encoder dlnetwork
%     vae_decoder.mat            - trained decoder dlnetwork
%     vae_embeddings.csv         - one row per call:
%                                    cohort, mat_file, call_id,
%                                    begin_s, end_s, z_0 ... z_{D-1}
%     vae_training_metadata.json - hyperparameters + cohort counts +
%                                  training duration
%     vae_elbo_curve.png         - ELBO loss curve (DS's training figure)
%     vae_sample_reconstructions.png - 16 input/recon pairs for inspection
%
%   Required handles fields (built internally as a mock):
%     handles.data.settings.detectionfolder
%     handles.data.settings.EntropyThreshold  (0.215, DS default)
%     handles.data.settings.AmplitudeThreshold (0.825, DS default)
%     handles.data.squeakfolder
%
%   Usage (5970 wild + lab_131204, per-cohort cage notch):
%     >> opts = struct();
%     >> opts.subsamplePerCohort       = 8000;
%     >> opts.cohorts                  = {'5970', 'lab_131204'};
%     >> opts.frequencyMaskKhzByCohort = struct( ...
%            'lab_131204', [50.4 51.0], ...   % canonical cage tonal line
%            'x5970',      []);                % wild: no cage notch
%     >> deepsqueak_train_vae( ...
%            '\\wsl.localhost\Ubuntu\home\shachar\projects\mickey_london_lab\deepsqueak_mats_combined', ...
%            'C:\path\to\DeepSqueak', ...
%            '\\wsl.localhost\Ubuntu\home\shachar\projects\mickey_london_lab\results\vae_5970_lab', ...
%            opts)
%
%     Or:
%       matlab -batch "deepsqueak_train_vae('matDir','dsDir','outDir',opts)"

    %% === Defaults ===
    if nargin < 4 || isempty(opts), opts = struct(); end
    opts = setDefault(opts, 'cohortFromFilename', '(5970|lab_131204|3452|9252)');
    opts = setDefault(opts, 'subsamplePerCohort', 8000);
    opts = setDefault(opts, 'frequencyMaskKhz', []);
    opts = setDefault(opts, 'frequencyMaskKhzByCohort', struct());
    opts = setDefault(opts, 'cohorts', {});
    opts = setDefault(opts, 'numEpochs', 250);
    opts = setDefault(opts, 'latentDim', 32);
    opts = setDefault(opts, 'miniBatchSize', 128);
    opts = setDefault(opts, 'learningRate', 2.5e-4);
    opts = setDefault(opts, 'randomSeed', 42);
    opts = setDefault(opts, 'freqRangeKhz', [15 75]);

    % IMAGE_SIZE is DS's VAE input dimension (from VAE_model.m:5),
    % NOT corpus.STFT_HOP. Coincident value, unrelated concept.
    IMAGE_SIZE = [128 128 1];
    ENTROPY_THRESHOLD = 0.215;
    AMPLITUDE_THRESHOLD = 0.825;

    %% === Banner ===
    fprintf('\n');
    fprintf('==================================================\n');
    fprintf(' DeepSqueak Headless VAE Training\n');
    fprintf('==================================================\n');
    fprintf('  Mat dir:               %s\n', matDir);
    fprintf('  DS folder:             %s\n', dsFolder);
    fprintf('  Output dir:            %s\n', outputDir);
    fprintf('  Cohort regex:          %s\n', opts.cohortFromFilename);
    fprintf('  Subsample per cohort:  %d\n', opts.subsamplePerCohort);
    fprintf('  Latent dim:            %d\n', opts.latentDim);
    fprintf('  Epochs:                %d\n', opts.numEpochs);
    fprintf('  Batch size:            %d\n', opts.miniBatchSize);
    fprintf('  Learning rate:         %g\n', opts.learningRate);
    fprintf('  Freq range:            [%g %g] kHz\n', opts.freqRangeKhz(1), opts.freqRangeKhz(2));
    if ~isempty(opts.frequencyMaskKhz)
        fprintf('  Global freq mask (kHz):');
        for k = 1:size(opts.frequencyMaskKhz,1)
            fprintf(' [%g %g]', opts.frequencyMaskKhz(k,1), opts.frequencyMaskKhz(k,2));
        end
        fprintf('\n');
    end
    if ~isempty(fieldnames(opts.frequencyMaskKhzByCohort))
        fprintf('  Per-cohort freq masks:\n');
        f = fieldnames(opts.frequencyMaskKhzByCohort);
        for k = 1:length(f)
            bands = opts.frequencyMaskKhzByCohort.(f{k});
            if isempty(bands)
                fprintf('    %s: (no mask)\n', f{k});
            else
                fprintf('    %s:', f{k});
                for kk = 1:size(bands, 1)
                    fprintf(' [%g %g]', bands(kk, 1), bands(kk, 2));
                end
                fprintf('\n');
            end
        end
    end
    if ~isempty(opts.cohorts)
        fprintf('  Cohort whitelist:      %s\n', strjoin(opts.cohorts, ', '));
    end
    fprintf('  Random seed:           %d\n', opts.randomSeed);
    fprintf('==================================================\n\n');

    %% === Input validation ===
    if ~isfolder(matDir)
        error('deepsqueak_train_vae:badMatDir', ...
            'matDir does not exist: %s', matDir);
    end
    if ~isfolder(dsFolder)
        error('deepsqueak_train_vae:badDsFolder', ...
            'dsFolder does not exist: %s', dsFolder);
    end
    dsFunc = fullfile(dsFolder, 'Functions');
    if ~isfolder(dsFunc)
        error('deepsqueak_train_vae:noDsFunctions', ...
            'DeepSqueak Functions/ folder not found at: %s', dsFunc);
    end
    vaeFunc = fullfile(dsFunc, 'Variational Autoencoder');
    if ~isfolder(vaeFunc)
        error('deepsqueak_train_vae:noVaeFunctions', ...
            ['DS Variational Autoencoder/ folder not found at: %s\n' ...
             'Required DS version: v3.1 (commit 1be0267) or newer.'], vaeFunc);
    end
    if ~isfolder(outputDir)
        mkdir(outputDir);
        fprintf('Created output directory: %s\n\n', outputDir);
    end

    %% === Path + headless setup ===
    addpath(genpath(dsFunc));
    fprintf('Added DeepSqueak functions to path.\n');
    set(0, 'DefaultFigureVisible', 'off');
    rng(opts.randomSeed);

    % Mock handles (3 fields needed by CreateFocusSpectrogram, verified
    % from deepsqueak_batch_classify.m source).
    handles = struct();
    handles.data.settings.detectionfolder = matDir;
    handles.data.settings.EntropyThreshold = ENTROPY_THRESHOLD;
    handles.data.settings.AmplitudeThreshold = AMPLITUDE_THRESHOLD;
    handles.data.squeakfolder = dsFolder;

    %% === Discover .mat files ===
    matFiles_struct = dir(fullfile(matDir, '*.mat'));
    if isempty(matFiles_struct)
        error('deepsqueak_train_vae:noMatFiles', ...
            'No .mat files found in: %s', matDir);
    end
    matFiles = {};
    for k = 1:length(matFiles_struct)
        fname = matFiles_struct(k).name;
        if startsWith(fname, 'Example ', 'IgnoreCase', true)
            continue;
        end
        matFiles{end+1} = fullfile(matDir, fname); %#ok<AGROW>
    end
    if isempty(matFiles)
        error('deepsqueak_train_vae:noMatFiles', ...
            'No project .mat files found in: %s', matDir);
    end
    fprintf('Found %d .mat detection files.\n\n', length(matFiles));

    %% === Phase A: Build spectrogram dataset (per-file loop) ===
    fprintf('--- Phase A: Building spectrogram dataset ---\n\n');

    allImages = {};
    allCohort = {};
    allMatFile = {};
    allCallId = [];
    allBeginS = [];
    allEndS = [];
    totalSkipped = 0;

    nWhitelistSkipped = 0;
    for j = 1:length(matFiles)
        matPath = matFiles{j};
        [~, matName, ~] = fileparts(matPath);
        cohort = parseCohort(matName, opts.cohortFromFilename);

        if ~isempty(opts.cohorts) && ~ismember(cohort, opts.cohorts)
            nWhitelistSkipped = nWhitelistSkipped + 1;
            continue;
        end

        fprintf('  [%d/%d] %s (cohort=%s) ... ', j, length(matFiles), matName, cohort);

        [Calls, audiodata_j] = loadCallfile(matPath, []);
        if isempty(Calls) || height(Calls) == 0
            fprintf('0 calls (skipped)\n');
            continue;
        end
        if ~isfile(audiodata_j.Filename)
            fprintf('\n    [WARN] Audio file missing: %s\n    SKIPPING file.\n', audiodata_j.Filename);
            continue;
        end

        audioReader = squeakData([]);
        audioReader.audiodata = audiodata_j;
        fileCount = 0;

        for i = 1:height(Calls)
            if Calls.Box(i,3) == 0 || Calls.Box(i,4) == 0
                totalSkipped = totalSkipped + 1;
                continue;
            end
            try
                % Same preprocessing chain as deepsqueak_batch_classify.m
                % lines 192-201 (the k-means wrapper). Output is a uint8
                % image at the call's natural aspect ratio; we resize to
                % 128x128 later in Phase B for VAE input.
                [I_raw, wind, noverlap, nfft, ~, ~, ~, ~, ~, ~, pow] = ...
                    CreateFocusSpectrogram(Calls(i,:), handles, true, [], audioReader);
                pow(pow == 0) = .01;
                pow = log10(pow);
                pow = rescale(imcomplement(abs(pow)));
                pow = flipud(pow);
                im = imadjust(pow, [.5 .9]);
                im_u8 = uint8(im .* 256);

                allImages{end+1, 1} = im_u8; %#ok<AGROW>
                allCohort{end+1, 1} = cohort; %#ok<AGROW>
                allMatFile{end+1, 1} = matName; %#ok<AGROW>
                allCallId(end+1, 1) = i; %#ok<AGROW>
                allBeginS(end+1, 1) = Calls.Box(i, 1); %#ok<AGROW>
                allEndS(end+1, 1) = Calls.Box(i, 1) + Calls.Box(i, 3); %#ok<AGROW>
                fileCount = fileCount + 1;
            catch ME
                totalSkipped = totalSkipped + 1;
                fprintf('\n    [WARN] Call %d failed: %s', i, ME.message);
            end
        end
        fprintf('%d calls\n', fileCount);
    end

    totalCalls = numel(allImages);
    if totalCalls == 0
        error('deepsqueak_train_vae:noCalls', 'No valid calls found.');
    end
    fprintf('\n  Total spectrograms built: %d  (skipped %d calls', totalCalls, totalSkipped);
    if nWhitelistSkipped > 0
        fprintf(', %d files filtered by cohort whitelist', nWhitelistSkipped);
    end
    fprintf(')\n\n');

    % Cohort counts
    [uniqueCohorts, ~, cohortIdx] = unique(allCohort);
    cohortCounts = accumarray(cohortIdx, 1);
    fprintf('  Cohort breakdown (all calls):\n');
    for k = 1:length(uniqueCohorts)
        fprintf('    %s: %d\n', uniqueCohorts{k}, cohortCounts(k));
    end
    fprintf('\n');

    %% === Phase B: Resize to 128x128, apply frequency mask ===
    fprintf('--- Phase B: Resizing + frequency mask ---\n\n');

    H = IMAGE_SIZE(1);
    W = IMAGE_SIZE(2);
    images = zeros(H, W, 1, totalCalls, 'uint8');
    for i = 1:totalCalls
        images(:, :, 1, i) = imresize(allImages{i}, [H W]);
    end
    fprintf('  Resized %d images to %dx%d.\n', totalCalls, H, W);

    % Per-cohort masking. For each cohort, look up its band matrix in
    % opts.frequencyMaskKhzByCohort (via matlab.lang.makeValidName to
    % handle cohort labels that start with digits like '5970'). Fall back
    % to opts.frequencyMaskKhz if the per-cohort entry is empty/missing.
    % This avoids the prior failure mode where the lab cage mask would
    % also zero those bands in wild 5970 images.

    % Snapshot pre-mask samples for the sanity-check render (Phase B.5).
    % Up to nSamplePerCohort indices per cohort, drawn before any mask
    % is applied so we can render before/after pairs.
    nSamplePerCohort = 4;
    sampleIdx_byCohort = cell(length(uniqueCohorts), 1);
    samplePre_byCohort = cell(length(uniqueCohorts), 1);
    for k = 1:length(uniqueCohorts)
        idxk = find(cohortIdx == k);
        nPick = min(nSamplePerCohort, numel(idxk));
        pick = idxk(randperm(numel(idxk), nPick));
        sampleIdx_byCohort{k} = pick;
        samplePre_byCohort{k} = images(:, :, 1, pick);
    end

    maskedAny = false;
    for k = 1:length(uniqueCohorts)
        thisCohort = uniqueCohorts{k};
        validName = matlab.lang.makeValidName(thisCohort);
        if isfield(opts.frequencyMaskKhzByCohort, validName)
            bandsK = opts.frequencyMaskKhzByCohort.(validName);
            maskSource = sprintf('frequencyMaskKhzByCohort.%s', validName);
        elseif ~isempty(opts.frequencyMaskKhz)
            bandsK = opts.frequencyMaskKhz;
            maskSource = 'frequencyMaskKhz (global)';
        else
            bandsK = [];
            maskSource = '';
        end

        idxk = find(cohortIdx == k);
        if isempty(bandsK)
            fprintf('  %s (%d images): no mask\n', thisCohort, numel(idxk));
            continue;
        end

        maskRows = khzBandsToRows(bandsK, H, opts.freqRangeKhz);
        fprintf('  %s (%d images, source=%s):\n', thisCohort, numel(idxk), maskSource);
        for kk = 1:size(maskRows, 1)
            r1 = maskRows(kk, 1); r2 = maskRows(kk, 2);
            % Note: linear-index slicing into a 4-D uint8 array is the
            % cheapest way to mask a subset of images without copying.
            images(r1:r2, :, 1, idxk) = 0;
            fprintf('    rows %d-%d  (kHz band [%g %g])\n', ...
                r1, r2, bandsK(kk, 1), bandsK(kk, 2));
            maskedAny = true;
        end
    end
    if ~maskedAny
        fprintf('  (no masking applied to any cohort)\n');
    end
    fprintf('\n');

    %% === Phase B.5: Sanity-check render (pre/post mask) ===
    fprintf('--- Phase B.5: Pre/post-mask sanity render ---\n\n');

    nRows = sum(cellfun(@numel, sampleIdx_byCohort));
    if nRows > 0
        sanityFig = figure('Color', 'w', 'Position', [100 100 1000 max(200, 80*nRows)], ...
            'Visible', 'off');
        tiledlayout(nRows, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
        for k = 1:length(uniqueCohorts)
            preImgs = samplePre_byCohort{k};
            idxs = sampleIdx_byCohort{k};
            for kk = 1:numel(idxs)
                nexttile;
                imagesc(preImgs(:, :, 1, kk)); axis off; colormap gray;
                title(sprintf('%s pre-mask  (call %d)', uniqueCohorts{k}, idxs(kk)), ...
                    'FontSize', 7, 'Interpreter', 'none');
                nexttile;
                imagesc(squeeze(images(:, :, 1, idxs(kk)))); axis off; colormap gray;
                title(sprintf('%s post-mask', uniqueCohorts{k}), ...
                    'FontSize', 7, 'Interpreter', 'none');
            end
        end
        sanityPath = fullfile(outputDir, 'vae_sample_masks.png');
        saveas(sanityFig, sanityPath);
        close(sanityFig);
        fprintf('  Saved pre/post-mask sanity render: %s\n\n', sanityPath);
    else
        fprintf('  (no images, skipping render)\n\n');
    end

    %% === Phase C: Stratified subsample + train/val split ===
    fprintf('--- Phase C: Stratified subsample + train/val split ---\n\n');

    trainIdx = [];
    valIdx = [];
    for k = 1:length(uniqueCohorts)
        thisCohort = uniqueCohorts{k};
        idx = find(cohortIdx == k);
        nThis = numel(idx);
        nKeep = min(nThis, opts.subsamplePerCohort);
        if nKeep < nThis
            idx = idx(randperm(nThis, nKeep));
        end
        % 90/10 split within cohort
        nTrain = round(0.9 * nKeep);
        perm = randperm(nKeep);
        trainIdx = [trainIdx; idx(perm(1:nTrain))]; %#ok<AGROW>
        valIdx   = [valIdx;   idx(perm(nTrain+1:end))]; %#ok<AGROW>
        fprintf('  %s: %d total -> %d train, %d val\n', ...
            thisCohort, nThis, nTrain, nKeep - nTrain);
    end
    fprintf('\n  Training pool: %d  Validation pool: %d\n\n', ...
        numel(trainIdx), numel(valIdx));

    %% === Phase D: Build dlarrays and train ===
    fprintf('--- Phase D: VAE training (%d epochs, batch %d, lr %g) ---\n\n', ...
        opts.numEpochs, opts.miniBatchSize, opts.learningRate);

    XTrain = dlarray(single(images(:, :, :, trainIdx)) ./ 256, 'SSCB');
    XTest  = dlarray(single(images(:, :, :, valIdx))   ./ 256, 'SSCB');

    if canUseGPU
        gpu = gpuDevice();
        fprintf('  GPU: %s (%.1f GB free)\n', gpu.Name, gpu.AvailableMemory / 1e9);
    else
        warning('No GPU detected. CPU training is impractical for this size.');
    end

    [encoderNet, decoderNet] = VAE_model();
    trainStart = tic;
    [encoderNet, decoderNet] = train_vae(encoderNet, decoderNet, XTrain, XTest);
    trainElapsed = toc(trainStart);
    fprintf('\n  Training complete (%.1f minutes).\n\n', trainElapsed / 60);

    % Save the ELBO curve figure that train_vae created
    elboFigs = findall(groot, 'Type', 'figure');
    if ~isempty(elboFigs)
        elboPath = fullfile(outputDir, 'vae_elbo_curve.png');
        saveas(elboFigs(end), elboPath);
        fprintf('  ELBO curve saved: %s\n', elboPath);
    end

    %% === Phase E: Save networks (UNC-safe) ===
    fprintf('\n--- Phase E: Saving networks ---\n\n');

    encPath = fullfile(outputDir, 'vae_encoder.mat');
    decPath = fullfile(outputDir, 'vae_decoder.mat');
    saveUncSafe(encPath, struct('encoderNet', encoderNet, ...
        'imageSize', IMAGE_SIZE, ...
        'latentDim', opts.latentDim, ...
        'freqRangeKhz', opts.freqRangeKhz, ...
        'frequencyMaskKhz', opts.frequencyMaskKhz));
    saveUncSafe(decPath, struct('decoderNet', decoderNet, ...
        'imageSize', IMAGE_SIZE, ...
        'latentDim', opts.latentDim));
    fprintf('  Encoder: %s\n  Decoder: %s\n\n', encPath, decPath);

    %% === Phase F: Extract embeddings for ALL calls ===
    fprintf('--- Phase F: Extracting embeddings for all %d calls ---\n\n', totalCalls);

    Xall = dlarray(single(images) ./ 256, 'SSCB');
    [~, zMean] = sampling(encoderNet, Xall);
    zMean = stripdims(zMean)';
    zMean = gather(extractdata(zMean));
    Z = double(zMean);
    fprintf('  Extracted %dx%d embedding matrix.\n\n', size(Z, 1), size(Z, 2));

    %% === Phase G: Sample reconstructions for inspection ===
    fprintf('--- Phase G: Reconstruction sample (16 calls) ---\n\n');

    nSample = min(16, totalCalls);
    sampleIdx = randperm(totalCalls, nSample);
    Xs = Xall(:, :, :, sampleIdx);
    [zS, ~] = sampling(encoderNet, Xs);
    Xrecon = sigmoid(forward(decoderNet, zS));
    Xs_arr = gather(extractdata(Xs));
    Xrecon_arr = gather(extractdata(Xrecon));

    reconFig = figure('Color', 'w', 'Position', [100 100 1200 600], 'Visible', 'off');
    tiledlayout(4, 8, 'Padding', 'compact', 'TileSpacing', 'compact');
    for k = 1:nSample
        nexttile;
        imagesc(squeeze(Xs_arr(:, :, 1, k))); axis off; colormap gray;
        title(sprintf('in %d (%s)', sampleIdx(k), allCohort{sampleIdx(k)}), 'FontSize', 7);
        nexttile;
        imagesc(squeeze(Xrecon_arr(:, :, 1, k))); axis off; colormap gray;
        title('recon', 'FontSize', 7);
    end
    reconPath = fullfile(outputDir, 'vae_sample_reconstructions.png');
    saveas(reconFig, reconPath);
    close(reconFig);
    fprintf('  Reconstruction sample saved: %s\n\n', reconPath);

    %% === Phase H: Write embeddings CSV ===
    fprintf('--- Phase H: Writing embeddings CSV ---\n\n');

    D = size(Z, 2);
    zNames = cell(1, D);
    for d = 0:D-1
        zNames{d+1} = sprintf('z_%d', d);
    end
    varNames = [{'cohort', 'mat_file', 'call_id', 'begin_s', 'end_s'}, zNames];

    T = cell2table([allCohort, allMatFile, num2cell(allCallId), ...
        num2cell(allBeginS), num2cell(allEndS), num2cell(Z)], ...
        'VariableNames', varNames);

    csvPath = fullfile(outputDir, 'vae_embeddings.csv');
    writetableUncSafe(T, csvPath);
    fprintf('  Embeddings written: %s\n\n', csvPath);

    %% === Phase I: Training metadata ===
    meta = struct();
    meta.timestamp = datestr(now, 'yyyy-mm-ddTHH:MM:SS');
    meta.matDir = matDir;
    meta.dsFolder = dsFolder;
    meta.outputDir = outputDir;
    meta.opts = opts;
    meta.totalCalls = totalCalls;
    meta.totalSkipped = totalSkipped;
    meta.cohortCounts = struct();
    for k = 1:length(uniqueCohorts)
        meta.cohortCounts.(matlab.lang.makeValidName(uniqueCohorts{k})) = cohortCounts(k);
    end
    meta.trainingSeconds = trainElapsed;
    meta.embeddingShape = size(Z);
    meta.dsSourceCommit = '1be0267';

    metaPath = fullfile(outputDir, 'vae_training_metadata.json');
    writeJsonUncSafe(metaPath, meta);
    fprintf('  Metadata: %s\n\n', metaPath);

    %% === Summary ===
    fprintf('==================================================\n');
    fprintf(' VAE Training Complete\n');
    fprintf('==================================================\n');
    fprintf('  Calls embedded: %d\n', totalCalls);
    fprintf('  Latent dim:     %d\n', D);
    fprintf('  Training time:  %.1f minutes\n', trainElapsed / 60);
    fprintf('  Outputs:\n');
    fprintf('    %s\n', encPath);
    fprintf('    %s\n', decPath);
    fprintf('    %s\n', csvPath);
    fprintf('    %s\n', metaPath);
    fprintf('    %s\n', reconPath);
    fprintf('==================================================\n');
    fprintf('\n  Next step (Python/WSL):\n');
    fprintf('    .venv/bin/python scripts/analyze_vae_embeddings.py \\\n');
    fprintf('        --embeddings %s \\\n', strrep(csvPath, '\', '/'));
    fprintf('        --out-dir results/vae_analysis/\n\n');

    set(0, 'DefaultFigureVisible', 'on');
end


%% ===== Local helper functions =====

function s = setDefault(s, field, value)
    if ~isfield(s, field) || isempty(s.(field))
        s.(field) = value;
    end
end

function cohort = parseCohort(name, pattern)
    tok = regexp(name, pattern, 'match', 'once');
    if isempty(tok)
        cohort = 'unknown';
    else
        cohort = tok;
    end
end

function rows = khzBandsToRows(bandsKhz, H, freqRangeKhz)
    % Map kHz bands to image rows for a 128x128 spectrogram with high
    % frequency at row 1 (after flipud in preprocessing).
    fLow = freqRangeKhz(1);
    fHigh = freqRangeKhz(2);
    N = size(bandsKhz, 1);
    rows = zeros(N, 2);
    for k = 1:N
        hLowKhz = bandsKhz(k, 1);
        hHighKhz = bandsKhz(k, 2);
        rowAtHigh = round((fHigh - hHighKhz) / (fHigh - fLow) * (H - 1) + 1);
        rowAtLow  = round((fHigh - hLowKhz)  / (fHigh - fLow) * (H - 1) + 1);
        r1 = max(1, min(H, min(rowAtHigh, rowAtLow)));
        r2 = max(1, min(H, max(rowAtHigh, rowAtLow)));
        rows(k, :) = [r1, r2];
    end
end

function saveUncSafe(targetPath, data)
    % MATLAB save() can fail on \\wsl.localhost\... UNC paths. Save to
    % tempdir, copy to target, delete temp. Pattern matches the existing
    % deepsqueak_batch_classify.m approach.
    [~, name, ext] = fileparts(targetPath);
    tmp = fullfile(tempdir, [name ext]);
    save(tmp, '-struct', 'data', '-v7.3');
    copyfile(tmp, targetPath);
    delete(tmp);
end

function writetableUncSafe(T, targetPath)
    [~, name, ext] = fileparts(targetPath);
    tmp = fullfile(tempdir, [name ext]);
    writetable(T, tmp);
    copyfile(tmp, targetPath);
    delete(tmp);
end

function writeJsonUncSafe(targetPath, s)
    [~, name, ext] = fileparts(targetPath);
    tmp = fullfile(tempdir, [name ext]);
    fid = fopen(tmp, 'w');
    fwrite(fid, jsonencode(s, 'PrettyPrint', true), 'char');
    fclose(fid);
    copyfile(tmp, targetPath);
    delete(tmp);
end
