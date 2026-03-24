"""Launch the Parts Finder API server."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from parts_finder.api.app import create_app
from parts_finder.config import AppConfig
import uvicorn

if __name__ == "__main__":
    _here = Path(__file__).resolve().parent
    config = AppConfig(db_path=str(_here / "parts_finder.db"))
    uvicorn.run(create_app(config), host="127.0.0.1", port=8000)
