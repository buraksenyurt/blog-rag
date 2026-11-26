import subprocess
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "ui" / "app.py"


def test_app_imports_without_project_root_on_path(tmp_path: Path) -> None:
    """`streamlit run ui/app.py` gibi: proje kökü sys.path'te değilken de import çalışmalı."""
    code = (
        "import runpy, sys; "
        f"root = {str(APP.parent.parent)!r}; "
        "sys.path = [p for p in sys.path if p not in ('', '.', root)]; "
        f"runpy.run_path({str(APP)!r}, run_name='import_check')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr[-2000:]
