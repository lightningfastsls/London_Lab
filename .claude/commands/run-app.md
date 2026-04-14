# Run the USV Detection App

Launch the PyQt6 desktop application for USV detection and review.

## Command
```bash
.venv/bin/python scripts/run_app.py
```

## Notes
- The app uses the production model at `models/hard_neg_retrain/best_model.pt`
- WAV files span multiple directories — no single canonical location
- Close the app window or use Ctrl+C in the terminal to stop

## Troubleshooting
If the app fails to start:
1. Check that .venv exists and has PyQt6 installed
2. Run `.venv/bin/pip install -r requirements.txt`
3. Check for syntax errors: `.venv/bin/python -m py_compile scripts/run_app.py`
4. Torch must be imported before PyQt6 (handled by `run_app.py`)
