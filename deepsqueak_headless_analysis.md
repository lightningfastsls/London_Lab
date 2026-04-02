# DeepSqueak Headless Classification: Source Code Analysis & Feasibility Report

*Based on reading every relevant source file in DrCoffey/DeepSqueak (v3.1, commit 1be0267)*

---

## Executive Summary: FEASIBLE

Headless batch classification is **fully feasible**. The critical insight from the source code analysis is that DeepSqueak's classification functions have a clean separation between GUI logic (file pickers, dialogs, interactive review) and computational logic (spectrogram generation, stats, clustering, neural network inference). Every computational function can be called directly with a minimal mock `handles` struct containing just **3 data fields**.

The recommended approach is **not** to call the existing `_Callback` wrappers headlessly, but to write a new `deepsqueak_batch_classify.m` that calls the underlying computational functions directly, bypassing all GUI code.

---

## 1. CreateClusteringData.m

- **Location**: `Functions/Call Classification/CreateClusteringData.m`
- **Signature**: `[ClusteringData, clustAssign, freqRange, maxDuration, spectrogramOptions] = CreateClusteringData(handles, varargin)`
- **Name-value params**: `'forClustering'`, `'spectrogramOptions'`, `'scale_duration'`, `'fixed_frequency'`, `'freqRange'`, `'save_data'`, `'for_denoise'`

### handles fields accessed
| Field | GUI or Data? | Purpose |
|-------|-------------|---------|
| `handles.data.settings.detectionfolder` | Data | Default path for `uigetfile` dialog |
| `handles.data.settings.EntropyThreshold` | Data | Passed to `CalculateStats` (line 125) |
| `handles.data.settings.AmplitudeThreshold` | Data | Passed to `CalculateStats` (line 125) |

Also passed **through** to:
- `loadCallfile(fullfile(filePath,fileName{j}), handles)` — but `handles` is only used there to find missing audio files. If your .mat files already contain valid `audiodata.Filename`, this is a no-op.
- `CreateFocusSpectrogram(Calls(i,:), handles, true, [], audioReader)` — when `make_spectrogram=true` (which this always passes), `handles` is **never touched** inside CreateFocusSpectrogram. It's only used when `make_spectrogram=false` (GUI page view mode).

### GUI blockers
- **Line 37**: `uigetfile(...)` — file selection dialog. Must be bypassed.
- **Line 43**: `waitbar(...)` — progress bar. Harmless headless (works silently).
- **Line 164**: `uiputfile(...)` — optional save dialog. Must be bypassed.

### Can run headless?
**Yes, with two modifications:**
1. Replace `uigetfile` with direct file path injection
2. Set `save_data=false` to skip `uiputfile`

### Key logic
Iterates over selected .mat detection files. For each file, loads Calls table + audiodata via `loadCallfile`. Creates a `squeakData` audio reader. For each call, calls `CreateFocusSpectrogram` to make spectrogram, then if `forClustering=true`, calls `CalculateStats` to extract contour features. Returns a table with columns: `Spectrogram`, `Box`, `MinFreq`, `Duration`, `xFreq`, `xTime`, `Filename`, `callID`, `Power`, `Bandwidth`.

### Mock handles needed
```matlab
handles.data.settings.detectionfolder = '/path/to/detections/';
handles.data.settings.EntropyThreshold = 0.215;   % default from squeakData
handles.data.settings.AmplitudeThreshold = 0.825;  % default from squeakData
```

---

## 2. CreateFocusSpectrogram.m

- **Location**: `Functions/CreateFocusSpectrogram.m`
- **Signature**: `[I,windowsize,noverlap,nfft,rate,box,s,fr,ti,audio,p] = CreateFocusSpectrogram(call, handles, make_spectrogram, options, audioReader)`

### handles fields accessed (only when make_spectrogram=false)
| Field | GUI or Data? | When used |
|-------|-------------|-----------|
| `handles.data.page_spect.s` | Data | Only when `make_spectrogram=false` (GUI page view) |
| `handles.data.page_spect.t` | Data | Only when `make_spectrogram=false` |
| `handles.data.page_spect.f` | Data | Only when `make_spectrogram=false` |

### Can run headless?
**Yes — handles is completely unused when make_spectrogram=true**, which is the path taken by both `CreateClusteringData` and `excel_Callback`. The function reads audio via `audioReader.AudioSamples(box(1), box(1) + box(3))` and computes spectrogram with MATLAB's `spectrogram()`.

### Key logic
Auto-calculates optimal FFT window size from call box dimensions if no options struct provided. Reads audio samples from the `squeakData` reader, computes spectrogram, and extracts the frequency-bounded sub-image.

---

## 3. CreateSpectrogram.m (legacy v2)

- **Location**: `Functions/CreateSpectrogram.m`
- **Signature**: `[I,windowsize,noverlap,nfft,rate,box,s,fr,ti,audio,AudioRange] = CreateSpectrogram(call)`

### handles fields accessed: NONE

This is the old v2 format function. It expects `call.Rate`, `call.Audio`, `call.RelBox` — columns that are **removed** by `loadCallfile` in v3 (line 40: `removevars(..., {'RelBox', 'Rate', 'Audio'})`). You won't use this function in v3.

---

## 4. CalculateStats.m

- **Location**: `Functions/CalculateStats.m`
- **Signature**: `stats = CalculateStats(I, windowsize, noverlap, nfft, SampleRate, Box, EntropyThreshold, AmplitudeThreshold, verbose)`

### handles fields accessed: NONE — fully standalone

### Can run headless? **YES, unconditionally**

### Key logic
This is the workhorse that computes the 16 acoustic features for the Excel export:

1. **Ridge detection**: Finds the brightest frequency at each time point, applies entropy + amplitude thresholds to select "real" contour points, smooth with `rlowess`
2. **Time stats**: BeginTime, EndTime, DeltaTime
3. **Frequency stats**: PrincipalFreq (median), LowFreq, HighFreq, DeltaFreq, stdev, PeakFreq
4. **Shape stats**: Slope (linear regression), Sinuosity (path length / straight-line distance)
5. **Power stats**: MeanPower (dB/Hz PSD), Power (per-point), SignalToNoise (mean 1-entropy on ridge)

### Output fields → Excel columns
| stats field | Excel column |
|------------|-------------|
| BeginTime | Begin Time (s) |
| EndTime | End Time (s) |
| DeltaTime | Call Length (s) |
| PrincipalFreq | Principal Frequency (kHz) |
| LowFreq | Low Freq (kHz) |
| HighFreq | High Freq (kHz) |
| DeltaFreq | Delta Freq (kHz) |
| stdev | Frequency Standard Deviation (kHz) |
| Slope | Slope (kHz/s) |
| Sinuosity | Sinuosity |
| MeanPower | Mean Power (dB/Hz) |
| SignalToNoise | Tonality |
| PeakFreq | Peak Freq (kHz) |

---

## 5. loadCallfile.m

- **Location**: `Functions/loadCallfile.m`
- **Signature**: `[Calls, audiodata, ClusteringData] = loadCallfile(filename, handles)`

### handles fields accessed (conditionally)
| Field | GUI or Data? | When used |
|-------|-------------|-----------|
| `handles.audiofilesnames` | Data | Only if audiodata is missing/invalid in .mat |
| `handles.audiofiles` | Data | Only if audiodata is missing/invalid |

### Can run headless?
**Yes, if your .mat files contain valid `audiodata` with a correct `Filename` path.** The audio-matching logic (lines 48-84) only triggers when:
- `handles` is not empty, AND
- `audiodata` is empty OR `audiodata.Filename` doesn't exist on disk

If `handles` is passed as `[]`, the function skips this block entirely (used by `UpdateCluster`).

### Critical detail for your pipeline
Your `create_deepsqueak_mats.m` must save a valid `audiodata` struct with `Filename` pointing to the actual WAV path. The `audiodata` struct should match `audioinfo()` output: `Filename`, `SampleRate`, `TotalSamples`, `Duration`, `NumChannels`, `BitsPerSample`, `CompressionMethod`, `Title`, `Comment`, `Artist`.

---

## 6. squeakData.m

- **Location**: `Functions/squeakData.m`
- **Class**: `squeakData < handle`

### Key properties
- `squeakfolder` — DeepSqueak installation root
- `settings` — loaded from `settings.mat`, includes all defaults
- `audiodata` — struct from `audioinfo()`, set per-file
- `defaultSettings` — hardcoded defaults including:
  - `EntropyThreshold = 0.215`
  - `AmplitudeThreshold = 0.825`
  - `detectionfolder`, `networkfolder`, `audiofolder`

### Key method: AudioSamples(startTime, finalTime)
Reads audio on-demand via `audioread(obj.audiodata.Filename, [start, stop])`. Caches samples to avoid redundant reads.

### For headless use
You create a `squeakData` instance via `audioReader = squeakData([])`, then set `audioReader.audiodata = yourAudiodataStruct`. This is exactly what `export_Calls.m` already does (line 31).

---

## 7. UnsupervisedClustering_Callback.m

- **Location**: `Functions/Call Classification/UnsupervisedClustering_Callback.m`
- **Signature**: `UnsupervisedClustering_Callback(hObject, eventdata, handles)`

### handles fields accessed
| Field | Via | GUI or Data? |
|-------|-----|-------------|
| `handles.data.settings.detectionfolder` | CreateClusteringData | Data |
| `handles.data.settings.EntropyThreshold` | CreateClusteringData → CalculateStats | Data |
| `handles.data.settings.AmplitudeThreshold` | CreateClusteringData → CalculateStats | Data |
| `handles.data.squeakfolder` | For model save/load paths | Data |

### GUI blockers (MANY)
- `questdlg` — method selection (ARTwarp / VAE+Contour / Contour Parameters)
- `questdlg` — "From existing model?"
- `inputdlg` — cluster parameter weights, ARTwarp settings, k optimization
- `uigetfile` / `uiputfile` — model file selection, save location
- `clusteringGUI(...)` — **full interactive review GUI** (lines 152)
- `figure` — montage of exemplars

### Can run headless?
**Not directly.** But the underlying computational functions it calls are all headless-compatible:
- `CreateClusteringData(handles, ...)` — needs mock handles + file path injection
- `get_kmeans_data(ClusteringData, slope_weight, freq_weight, duration_weight)` — standalone
- `kmeans_opt(data, maxK, cutoff, repeats)` — creates figure but computes fine headlessly
- `knnsearch(C, data, 'Distance', 'euclidean')` — MATLAB builtin
- `ARTwarp2(ClusteringData.xFreq, settings)` — standalone
- `clusteringGUI(...)` — **must be skipped** (auto-accept all clusters)
- `UpdateCluster(ClusteringData, clustAssign, clusterName, rejected)` — standalone

### Three clustering paths

**Path A: Contour Parameters (k-means on contour features)**
```
CreateClusteringData → get_kmeans_data → kmeans_opt/kmeans → knnsearch → UpdateCluster
```
Parameters: shape_weight=3, freq_weight=2, duration_weight=1 (defaults)

**Path B: Auto Encoder + Contour (VAE, recommended)**
```
create_VAE_model → extract_VAE_embeddings + freq contours → kmeans → knnsearch → UpdateCluster
```
Trains a VAE on 128×128 spectrograms, uses latent embeddings + frequency contours for k-means.

**Path C: ARTwarp**
```
CreateClusteringData → ARTwarp2(xFreq, settings) → GetARTwarpClusters → UpdateCluster
```
Settings: MatchThresh=5, CombineVigilance=2.5, OutlierThresh=8, LR=0.001, Iters=5, shape/freq/time importance=4/1/1

---

## 8. kmeans_opt.m

- **Location**: `Functions/Call Classification/kmeans_opt.m`
- **Signature**: `[IDX, C, SUMD, K] = kmeans_opt(X, maxK, cutoff, repeats)`

### GUI elements
- Creates a `figure` with elbow plot (line 42). Non-blocking headless — figure won't display but won't crash.

### Can run headless? **YES**

### Key logic
Runs k-means from k=1 to maxK, collects within-cluster distances, normalizes by k=1 distance, finds elbow via `knee_pt()`. Returns optimal cluster assignments and centroids.

---

## 9. ARTwarp2.m

- **Location**: `Functions/Call Classification/ARTwarp2.m`
- **Signature**: `[net, clustAssign] = ARTwarp2(LineData, settings)`

### GUI elements
- `waitbar` only (harmless headless)

### Can run headless? **YES**

### Key logic
Adaptive Resonance Theory with Dynamic Time Warping. Resizes all contours to 100 points, z-scores lengths and frequencies, iteratively assigns to nearest category or creates new category. Uses DTW distance scaled by importance weights. Combines similar categories periodically.

---

## 10. UpdateCluster.m

- **Location**: `Functions/Call Classification/UpdateCluster.m`
- **Signature**: `UpdateCluster(ClusteringData, clustAssign, clusterName, rejected)`

### handles fields accessed: **NONE**

### Can run headless? **YES, unconditionally**

### Key logic
Iterates over unique filenames in ClusteringData. For each file, loads it via `loadCallfile(file, [])`, updates `Calls.Type` with cluster names, sets `Calls.Accept = 0` for rejected/Noise calls, saves back with `save(file, 'Calls', '-append')`.

---

## 11. SupervisedClassification_Callback.m

- **Location**: `Functions/Call Classification/SupervisedClassification_Callback.m`
- **Signature**: `SupervisedClassification_Callback(hObject, eventdata, handles)`

### handles fields accessed
Same as CreateClusteringData, plus:
| Field | Purpose |
|-------|---------|
| `handles.data.squeakfolder` | Classifier model path |
| `handles.data.settings.detectionfolder` | Via CreateClusteringData |

### GUI blockers
- `uigetfile` — classifier network selection
- `questdlg` — "Update files?"
- `waitbar` — progress (harmless)

### Can run headless?
**Yes, trivially.** The core logic is just:
```matlab
net = load(modelPath, 'ClassifyNet', 'wind', 'noverlap', 'nfft', 'imageSize');
% ... create spectrograms via CreateClusteringData ...
images = resize_all_to_128x128(ClusteringData.Spectrogram);
for j = 1:N
    X = images(:,:,:,j) ./ 256;
    [Cl, sc] = classify(net.ClassifyNet, X);
    clustAssign(j,1) = Cl;
end
UpdateCluster(ClusteringData, clustAssign, clusterName, zeros(...));
```

### Network format
The classifier .mat contains:
- `ClassifyNet` — trained MATLAB `SeriesNetwork` or `DAGNetwork`
- `imageSize` — `[128, 128, 1]`
- `layers` — network layer array (for reference, not needed at inference)

### CNN architecture (from TrainSupervisedClassifier_Callback.m)
```
imageInputLayer [128, 128, 1]
conv2d(3, 16, stride=2) → BN → ReLU → maxPool(2,stride=2)
conv2d(5, 16, stride=1) → BN → ReLU → maxPool(2,stride=2)
conv2d(5, 32, stride=1) → BN → ReLU → maxPool(2,stride=2)
conv2d(5, 32, stride=1) → BN → ReLU
FC(32) → BN → ReLU
FC(num_categories) → softmax → classification
```
Input: 128×128 grayscale spectrogram, values in [0,1].

---

## 12. excel_Callback.m / export_Calls.m

- **Location**: `Functions/Import and Export/excel_Callback.m`, `Functions/Import and Export/export_Calls.m`

### handles fields accessed
| Field | Via | Purpose |
|-------|-----|---------|
| `handles.data.settings.detectionfolder` | export_Calls line 7 | File picker default |
| `handles.data.settings.EntropyThreshold` | excel_Callback line 21 | CalculateStats |
| `handles.data.settings.AmplitudeThreshold` | excel_Callback line 21 | CalculateStats |

### GUI blockers (in export_Calls.m)
- `uigetfile` — file selection
- `questdlg` — include rejected? merge files?
- `uigetdir` — output folder

### Can run headless?
**Write your own export — it's trivial.** The core loop in `excel_Callback` is:
```matlab
for each call:
    [I,windowsize,noverlap,nfft,rate,box] = CreateFocusSpectrogram(Calls(i,:), handles, true, [], audioReader);
    stats = CalculateStats(I,windowsize,noverlap,nfft,rate,box, EntropyThreshold, AmplitudeThreshold);
    % append stats to table
end
writetable(t, outputName);
```

### Excel output columns (18 total)
File, ID, Label, Accepted, Score, Begin Time (s), End Time (s), Call Length (s), Principal Frequency (kHz), Low Freq (kHz), High Freq (kHz), Delta Freq (kHz), Frequency Standard Deviation (kHz), Slope (kHz/s), Sinuosity, Mean Power (dB/Hz), Tonality, Peak Freq (kHz)

---

## 13. Networks/ Directory

### Contents (all detection networks, NOT classifiers)
| File | Type |
|------|------|
| `2025.02.18.Mouse.YoloR3.mat` | Mouse USV detector (YOLO v2) |
| `Mouse Detector YOLO R2.mat` | Mouse USV detector (older) |
| `2025.02.04.Rat.YoloR2.mat` | Rat USV detector |
| `2025.02.05.Rat.Long.YoloR2.mat` | Long rat call detector |
| `Rat Detector YOLO R1.mat` | Rat detector (older) |
| `Long Rat Detector YOLO R1.mat` | Long rat detector (older) |

**No pre-trained USV classifiers ship with DeepSqueak.** The `Clustering Models/` directory is created by users after training. The CHANGELOG mentions "a newly trained supervised classifier based on Wright et al.'s rat USV categories" was included in v2.6, but it's not in the current v3.1 repo.

### Network .mat internal structure
- `None` (opaque MATLAB object — the YOLO network)
- `imLength` — input image size
- `imScale` — scaling function
- `wind`, `noverlap`, `nfft` — spectrogram FFT parameters
- `version` — network version

---

## 14. clusteringGUI.m

- **Location**: `Functions/Call Classification/clusteringGUI.m`
- **Class**: `clusteringGUI < handle` (402 lines)
- **Purpose**: Interactive GUI for reviewing and accepting/rejecting/merging clusters

### Can run headless? **NO — and you don't need it**

This is the interactive cluster review window. For batch processing, you skip this entirely and auto-accept all cluster assignments, or apply your own filtering logic.

---

## 15. Variational Autoencoder (VAE)

- **Location**: `Functions/Variational Autoencoder/`

### create_VAE_model.m
Calls `CreateClusteringData` with `scale_duration=true, fixed_frequency=true`, resizes all spectrograms to 128×128, trains encoder/decoder with `train_vae`. Returns trained networks.

### extract_VAE_embeddings.m
Passes images through encoder to get latent embeddings. Returns embedding matrix.

### VAE_model.m
Defines the encoder and decoder architectures using `dlnetwork`.

### Can run headless?
The VAE training itself is headless-compatible (uses `dlarray` training loop). But `create_VAE_model` calls `CreateClusteringData` (same GUI issues as above) and a `figure; montage(...)` (line 17, wrapped in try-catch).

---

## 16. DeepSqueak.m — handles Initialization

### How handles.data is created (DeepSqueak_OpeningFcn, line 120)
```matlab
squeakfolder = fileparts(mfilename('fullpath'));  % DeepSqueak install dir
handles.data = squeakData(squeakfolder);          % Creates the data class
```
Then `update_folders` calls `handles.data.loadSettings()` which loads `settings.mat` from `squeakfolder`. If no settings file exists, it creates one from `defaultSettings`.

### The minimum mock for headless classification
```matlab
% Create squeakData with DeepSqueak path for settings access
handles.data = squeakData('/path/to/DeepSqueak');
handles.data.loadSettings();  % loads or creates settings.mat

% Or bypass squeakData entirely with a minimal struct:
handles.data.settings.detectionfolder = '/path/to/your/detections/';
handles.data.settings.EntropyThreshold = 0.215;
handles.data.settings.AmplitudeThreshold = 0.825;
handles.data.squeakfolder = '/path/to/DeepSqueak';
```

---

## 17. Recommended Headless Architecture

### Option A: Unsupervised Clustering (Contour k-means)

```matlab
function deepsqueak_batch_cluster(matFiles, dsFolder, outputExcel)
% HEADLESS UNSUPERVISED CLUSTERING
% matFiles: cell array of .mat detection file paths
% dsFolder: path to DeepSqueak installation
% outputExcel: output .xlsx path

    addpath(genpath(fullfile(dsFolder, 'Functions')));

    % Mock handles
    handles.data.settings.detectionfolder = fileparts(matFiles{1});
    handles.data.settings.EntropyThreshold = 0.215;
    handles.data.settings.AmplitudeThreshold = 0.825;
    handles.data.squeakfolder = dsFolder;

    % --- Step 1: Build ClusteringData (bypass uigetfile) ---
    ClusteringData = {};
    clustAssign = [];
    audioReader = squeakData([]);

    for j = 1:length(matFiles)
        [Calls, audiodata_j] = loadCallfile(matFiles{j}, []);
        if isempty(Calls); continue; end
        audioReader.audiodata = audiodata_j;

        for i = 1:height(Calls)
            [I,wind,noverlap,nfft,rate,box,s,fr,ti,~,pow] = ...
                CreateFocusSpectrogram(Calls(i,:), handles, true, [], audioReader);

            pow(pow==0) = .01;
            pow = log10(pow);
            pow = rescale(imcomplement(abs(pow)));
            pow = flipud(pow);
            im = imadjust(pow, [.5 .9]);

            stats = CalculateStats(I,wind,noverlap,nfft,rate,box, ...
                handles.data.settings.EntropyThreshold, ...
                handles.data.settings.AmplitudeThreshold);

            spectrange = audioReader.audiodata.SampleRate / 2000;
            FreqScale = spectrange / (1 + floor(nfft / 2));
            TimeScale = (wind - noverlap) / audioReader.audiodata.SampleRate;
            xFreq = FreqScale * stats.ridgeFreq_smooth + Calls.Box(i,2);
            xTime = stats.ridgeTime * TimeScale;

            ClusteringData = [ClusteringData
                [{uint8(im.*256)} {box} {box(2)} {stats.DeltaTime} ...
                 {xFreq} {xTime} {matFiles{j}} {i} {stats.Power} {box(4)}]'];
            clustAssign = [clustAssign; Calls.Type(i)];
        end
    end

    ClusteringData = cell2table(ClusteringData(:,1:10), ...
        'VariableNames', {'Spectrogram','Box','MinFreq','Duration', ...
        'xFreq','xTime','Filename','callID','Power','Bandwidth'});

    % --- Step 2: K-means clustering ---
    slope_weight = 3; freq_weight = 2; duration_weight = 1;
    % (inline get_kmeans_data logic)
    ReshapedX = cell2mat(cellfun(@(x) imresize(x',[1 13]), ...
        ClusteringData.xFreq, 'UniformOutput', false));
    slope = zscore(diff(ReshapedX,1,2));
    freq = zscore(cell2mat(cellfun(@(x) imresize(x',[1 12]), ...
        ClusteringData.xFreq, 'UniformOutput', false)));
    duration = zscore(repmat(ClusteringData.Duration,[1 12]));
    data = [freq.*freq_weight, slope.*slope_weight, duration.*duration_weight];

    [~, C] = kmeans_opt(data, min(100, size(data,1)), 0, 3);
    [clustAssign_new, ~] = knnsearch(C, data, 'Distance', 'euclidean');
    clusterName = categorical(1:size(C,1));

    % --- Step 3: Save clusters ---
    rejected = zeros(1, height(ClusteringData));
    clustAssign_cat = categorical(clustAssign_new);
    UpdateCluster(ClusteringData, clustAssign_cat, clusterName, rejected);

    % --- Step 4: Export to Excel ---
    % (see export logic below)
    export_stats_headless(matFiles, handles, outputExcel);
end
```

### Option B: Supervised Classification

```matlab
function deepsqueak_batch_supervised(matFiles, dsFolder, classifierPath, outputExcel)
% HEADLESS SUPERVISED CLASSIFICATION
    addpath(genpath(fullfile(dsFolder, 'Functions')));

    % Load classifier
    net = load(classifierPath, 'ClassifyNet', 'imageSize');
    imageSize = [128, 128, 1];

    % Mock handles
    handles.data.settings.detectionfolder = fileparts(matFiles{1});
    handles.data.settings.EntropyThreshold = 0.215;
    handles.data.settings.AmplitudeThreshold = 0.825;
    handles.data.squeakfolder = dsFolder;

    % Build ClusteringData (same as above but with scale_duration/fixed_freq)
    % ... [same loop as Option A, with scale_duration modifications] ...

    % Classify
    images = zeros([imageSize, height(ClusteringData)]);
    for i = 1:height(ClusteringData)
        images(:,:,:,i) = imresize(ClusteringData.Spectrogram{i}, imageSize(1:2));
    end

    clustAssign = categorical([]);
    for j = 1:height(ClusteringData)
        X = images(:,:,:,j) ./ 256;
        [Cl, ~] = classify(net.ClassifyNet, X);
        clustAssign(j,1) = Cl;
    end

    clusterName = unique(clustAssign);
    UpdateCluster(ClusteringData, clustAssign, clusterName, zeros(1, height(ClusteringData)));
end
```

### Option C: Headless Excel Export (standalone)

```matlab
function export_stats_headless(matFiles, handles, outputExcel)
% Export stats without GUI
    exceltable = {'File','ID','Label','Accepted','Score', ...
        'Begin Time (s)','End Time (s)','Call Length (s)', ...
        'Principal Frequency (kHz)','Low Freq (kHz)','High Freq (kHz)', ...
        'Delta Freq (kHz)','Frequency Standard Deviation (kHz)', ...
        'Slope (kHz/s)','Sinuosity','Mean Power (dB/Hz)','Tonality','Peak Freq (kHz)'};

    for j = 1:length(matFiles)
        audioReader = squeakData([]);
        [Calls, audioReader.audiodata] = loadCallfile(matFiles{j}, []);
        if isempty(Calls); continue; end

        for i = 1:height(Calls)
            if Calls.Box(i,3)==0 || Calls.Box(i,4)==0; continue; end

            [I,windowsize,noverlap,nfft,rate,box] = ...
                CreateFocusSpectrogram(Calls(i,:), handles, true, [], audioReader);
            stats = CalculateStats(I,windowsize,noverlap,nfft,rate,box, ...
                handles.data.settings.EntropyThreshold, ...
                handles.data.settings.AmplitudeThreshold);

            exceltable = [exceltable; ...
                {matFiles{j}, i, Calls.Type(i), Calls.Accept(i), Calls.Score(i), ...
                 stats.BeginTime, stats.EndTime, stats.DeltaTime, ...
                 stats.PrincipalFreq, stats.LowFreq, stats.HighFreq, ...
                 stats.DeltaFreq, stats.stdev, stats.Slope, stats.Sinuosity, ...
                 stats.MeanPower, stats.SignalToNoise, stats.PeakFreq}];
        end
    end

    t = cell2table(exceltable);
    writetable(t, outputExcel, 'WriteVariableNames', false);
end
```

---

## 18. GUI Obstacles Summary

| Function | GUI blockers | Headless strategy |
|----------|-------------|-------------------|
| `CreateClusteringData` | `uigetfile`, `uiputfile` | Pass file paths directly, set `save_data=false` |
| `UnsupervisedClustering_Callback` | `questdlg` ×4, `inputdlg` ×3, `uigetfile`, `uiputfile`, `clusteringGUI` | Call underlying functions directly |
| `SupervisedClassification_Callback` | `uigetfile`, `questdlg` | Load network directly, call `classify()` |
| `excel_Callback` / `export_Calls` | `uigetfile`, `questdlg` ×2, `uigetdir` | Write own export loop |
| `kmeans_opt` | `figure` (non-blocking) | Works headless — figure just won't display |
| `ARTwarp2` | `waitbar` | Works headless |
| `UpdateCluster` | `waitbar` | Works headless |
| `clusteringGUI` | Full GUI class | Skip entirely — auto-accept |

---

## 19. Critical Path Dependencies

```
Your Python pipeline
  → create_deepsqueak_mats.m  (Raven TSV → .mat with Calls + audiodata)
  → deepsqueak_batch_classify.m  (headless MATLAB script)
      ├─ loadCallfile()           [no GUI needed if audiodata is valid]
      ├─ squeakData()             [audio reader — no GUI]
      ├─ CreateFocusSpectrogram() [no GUI when make_spectrogram=true]
      ├─ CalculateStats()         [fully standalone]
      ├─ kmeans_opt() or classify() [computational only]
      ├─ UpdateCluster()          [fully standalone]
      └─ writetable()            [MATLAB builtin]
  → Your Python pipeline reads the .xlsx output
```

---

## 20. Important Implementation Notes

1. **audiodata.Filename paths**: Must be valid Windows paths accessible from MATLAB. Your UNC paths (`\\wsl$\Ubuntu\...`) should work, but test with a simple `audioinfo()` call first.

2. **MATLAB version**: DeepSqueak v3 requires MATLAB 2020a+. The `spectrogram` call signature and `dlarray`/`dlnetwork` APIs changed in 2020a.

3. **Required toolboxes**: Deep Learning Toolbox, Image Processing Toolbox, Signal Processing Toolbox, Statistics and Machine Learning Toolbox (for `kmeans`, `knnsearch`), Curve Fitting Toolbox (optional, for smoothing).

4. **The `call.Box` format**: `[start_time_s, low_freq_kHz, duration_s, bandwidth_kHz]`. This is the Position-style `[x, y, width, height]` format.

5. **Spectrogram image format**: `uint8` values 0-255, stored as `ClusteringData.Spectrogram{i}`. For supervised classification, resized to 128×128 and divided by 256 to get [0,1] range.

6. **No pre-trained mouse USV classifier exists** in the repo. You would need to either:
   - Train one using `TrainSupervisedClassifier_Callback` (needs labeled data)
   - Use unsupervised clustering to create categories first, then train
   - Use an external classifier (e.g., BootSnap, which you've already identified)

7. **The VAE path** (recommended by DeepSqueak) is the most sophisticated but requires training a VAE on your data first. For batch processing, this means a two-phase approach: (1) train VAE interactively once, (2) apply saved model headlessly.
