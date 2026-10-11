#!/usr/bin/env python3
"""Open the configured FinOlimp Reconcile workspace in OpenCode."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
AI_ROOT = ROOT / "stage3_ai"
AI_CONFIG = AI_ROOT / ".opencode" / "opencode.json"


def main() -> int:
    if not AI_CONFIG.is_file():
        print(f"Не найдена конфигурация FinOlimp OpenCode: {AI_CONFIG}", file=sys.stderr)
        return 2
    user_install = None
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        user_install = Path(os.environ["LOCALAPPDATA"]) / "Programs" / "OpenCode" / "opencode.exe"
    executable = str(user_install) if user_install and user_install.is_file() else shutil.which("opencode")
    if executable is None:
        print(
            "Команда opencode не найдена. Сначала запустите install_windows.ps1 "
            "или install_ubuntu.sh, затем откройте новый терминал.",
            file=sys.stderr,
        )
        return 127

    print("Открываю OpenCode в stage3_ai с агентом FinOlimp Reconcile.", flush=True)
    print("Сверка начнётся только после вашего запроса в чате.", flush=True)
    completed = subprocess.run(
        [executable, *sys.argv[1:]],
        cwd=AI_ROOT,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
