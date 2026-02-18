Picking up from earlier:
- The Streamlit app must be launched with `streamlit run scripts/usv_parameter_lab.py` (not `python ...`).
- In the sidebar, select a real `.wav` file or paste a full path; empty path defaults to `.` and fails.
- Docs for usage live in `README.md` under "USV Parameter Lab (Streamlit)".
- Completeness check vs task brief: implemented segment-only reads, baseline/variant compare with shared scaling, heuristic overlay + metrics, timing, sweep export with images + report.
- Likely gaps: sweep report is Markdown only (brief asked for Markdown/HTML) and there is JSON export but no YAML.

Tell me which you want next:
1) Fix the app UX for empty/invalid WAV path.
2) Add HTML + YAML exports to sweep.
3) Both.
You can run it in the same terminal (PowerShell) you used before, but use the Streamlit command instead of running the Python file directly.

From the repo root (`D:\mickey_london_lab`), run:

streamlit run scripts/usv_parameter_lab.py

If you prefer the venv?s Streamlit explicitly:

.\.venv\Scripts\streamlit.exe run scripts\usv_parameter_lab.py

Either way, Streamlit will open a browser tab with the app.
I can set the default WAV folder in this repo (e.g., in `src/usv_spectrogram/param_lab/app.py`) to `D:\mickey_london_lab\5970 USV`. That will apply to this project only.

If you want it for *all* of your projects, that?s outside this repo. I?d need to know where you want a shared default to live (e.g., a global config file, environment variable, or a shared .codex skill).

Do you want me to:
1) Update this repo only.
2) Set up a cross-project default (tell me how you want it stored).
---
Noted. I will separate entries in this file with `---` from now on.
