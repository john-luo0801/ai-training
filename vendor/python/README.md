# Bundled Python packages

This folder holds the wheel files for everything listed in
`requirements.txt`, plus PyInstaller (used by `build_exe.bat`).

Install them **without the internet** with:

```powershell
py -3 -m pip install --no-index --find-links vendor\python -r requirements.txt
```

Or double-click `install_deps.bat` in the project root.

To refresh the wheels after changing `requirements.txt` (needs internet once):

```powershell
py -3 data\download_vendor_packages.py
```

> These wheels target **Windows 64-bit + Python 3.14**, which matches the
> machine this project was packaged on. On another Python version or OS,
> re-run the download script so the correct wheels are fetched.
