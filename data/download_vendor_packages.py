"""Download wheels for requirements.txt (+ PyInstaller) into vendor/python."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "python"
REQUIREMENTS = ROOT / "requirements.txt"


def main() -> None:
    VENDOR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "-r",
        str(REQUIREMENTS),
        "pyinstaller",
        "colorama",
        "-d",
        str(VENDOR),
    ]
    print("Downloading into", VENDOR)
    subprocess.check_call(command)
    wheels = sorted(VENDOR.glob("*.whl"))
    print(f"Saved {len(wheels)} wheel file(s).")


if __name__ == "__main__":
    main()
