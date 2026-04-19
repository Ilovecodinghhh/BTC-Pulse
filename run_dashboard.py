#!/usr/bin/env python3
"""
BTC-Pulse Dashboard Launcher
Starts the Streamlit dashboard.
"""

import subprocess
import sys
from pathlib import Path

dashboard_path = Path(__file__).resolve().parent / "dashboard" / "app.py"

if __name__ == "__main__":
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", "8501",
        "--theme.base", "dark",
    ])
