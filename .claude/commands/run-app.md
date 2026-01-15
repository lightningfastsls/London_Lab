# Run the USV Parameter Lab App

Launch the Streamlit application for interactive spectrogram exploration.

## Command
```powershell
.\.venv\Scripts\streamlit.exe run scripts/usv_parameter_lab.py
```

## Notes
- The app will open in your default browser
- Make sure USV_WAV_DIR is set or WAV files are in `<repo>/5970 USV`
- Use Ctrl+C in the terminal to stop the app

## Troubleshooting
If the app fails to start:
1. Check that .venv exists and has streamlit installed
2. Run `.\.venv\Scripts\pip.exe install -r requirements.txt`
3. Check for syntax errors: `.\.venv\Scripts\python.exe -m py_compile scripts/usv_parameter_lab.py`
