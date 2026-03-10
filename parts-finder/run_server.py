"""Launch the Parts Finder API server."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from parts_finder.api.app import create_app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)
