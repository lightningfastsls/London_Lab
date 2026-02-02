"""Test which import causes the GUI app to fail."""

import sys
from pathlib import Path

print("Step 1: Testing basic imports...")
try:
    import torch
    print(f"  [OK] torch {torch.__version__}")
except Exception as e:
    print(f"  [FAIL] torch failed: {e}")
    sys.exit(1)

try:
    from PyQt6.QtWidgets import QApplication
    print(f"  [OK] PyQt6.QtWidgets")
except Exception as e:
    print(f"  [FAIL] PyQt6 failed: {e}")
    sys.exit(1)

print("\nStep 2: Testing torch AFTER PyQt6...")
try:
    import torch as torch2
    print(f"  [OK] torch still works after PyQt6")
except Exception as e:
    print(f"  [FAIL] torch failed after PyQt6: {e}")
    sys.exit(1)

print("\nStep 3: Testing app imports...")
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from usv_spectrogram.app.core.audio_loader import AudioLoader
    print(f"  [OK] AudioLoader")
except Exception as e:
    print(f"  [FAIL] AudioLoader failed: {e}")
    sys.exit(1)

try:
    from usv_spectrogram.app.core.sliding_inference import SlidingInference
    print(f"  [OK] SlidingInference")
except Exception as e:
    print(f"  [FAIL] SlidingInference failed: {e}")
    sys.exit(1)

try:
    from usv_spectrogram.app.main_window import MainWindow
    print(f"  [OK] MainWindow")
except Exception as e:
    print(f"  [FAIL] MainWindow failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[OK] All imports successful! The app should work.")
print("\nStep 4: Testing QApplication creation...")
try:
    app = QApplication([])
    print(f"  [OK] QApplication created")
except Exception as e:
    print(f"  [FAIL] QApplication failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[SUCCESS] Everything works! GUI app should launch.")
