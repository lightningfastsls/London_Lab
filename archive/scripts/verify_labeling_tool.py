"""Quick verification script for labeling tool syntax."""

from pathlib import Path
import sys

# Add src to path
REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

print("Verifying labeling tool files...")

# Test imports
try:
    from usv_spectrogram.labeling import labeling_app
    print("✓ labeling_app.py imports successfully")
except Exception as e:
    print(f"✗ labeling_app.py import failed: {e}")
    sys.exit(1)

# Check functions exist
required_functions = [
    "load_candidates",
    "load_existing_labels",
    "save_label",
    "get_spectrogram_path",
    "initialize_session_state",
    "render_navigation",
    "render_candidate_info",
    "render_labeling_controls",
    "render_labeling_guide",
    "run",
]

for func_name in required_functions:
    if hasattr(labeling_app, func_name):
        print(f"✓ Function '{func_name}' exists")
    else:
        print(f"✗ Function '{func_name}' missing")
        sys.exit(1)

# Check required files
required_files = [
    REPO_ROOT / "src" / "usv_spectrogram" / "labeling" / "__init__.py",
    REPO_ROOT / "src" / "usv_spectrogram" / "labeling" / "labeling_app.py",
    REPO_ROOT / "src" / "usv_spectrogram" / "labeling" / "README.md",
    REPO_ROOT / "scripts" / "usv_labeling_tool.py",
]

for filepath in required_files:
    if filepath.exists():
        print(f"✓ File exists: {filepath.name}")
    else:
        print(f"✗ File missing: {filepath}")
        sys.exit(1)

# Check expected input files
input_files = [
    REPO_ROOT / "candidates_optimized.csv",
    REPO_ROOT / "spectrograms_review",
]

for filepath in input_files:
    if filepath.exists():
        print(f"✓ Input exists: {filepath.name}")
    else:
        print(f"⚠ Input missing (expected): {filepath.name}")

print("\n✅ All verification checks passed!")
print("\nTo run the labeling tool:")
print("  .\.venv\Scripts\streamlit.exe run scripts/usv_labeling_tool.py")
