#!/usr/bin/env python3
"""Hook PostToolUse: después de editar código Python, corre los tests rápidos.

Si fallan, sale con código 2 y el agente recibe el error para corregirlo.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path(__file__).resolve().parents[2]))


def python_del_proyecto():
    for candidato in (RAIZ / ".venv/bin/python", RAIZ / "venv/bin/python",
                      RAIZ / ".venv/Scripts/python.exe", RAIZ / "venv/Scripts/python.exe"):
        if candidato.exists():
            return str(candidato)
    return sys.executable


def main():
    try:
        evento = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)

    entrada = evento.get("tool_input", {}) or {}
    ruta = (entrada.get("file_path") or "").replace("\\", "/")
    if not ruta.endswith(".py") or not any(d in ruta for d in ("backend/", "training/", "tests/")):
        sys.exit(0)

    carpetas = [c for c in ("verificacion/tests", "tests") if (RAIZ / c).is_dir()]
    resultado = subprocess.run(
        [python_del_proyecto(), "-m", "pytest", *carpetas,
         "-m", "not hallazgo and not lento", "-q", "-x", "--no-header", "-p", "no:cacheprovider"],
        cwd=RAIZ, capture_output=True, text=True, timeout=600,
    )
    # Código 5 = no se encontraron tests (por ejemplo, tests/ aún no existe)
    if resultado.returncode in (0, 5):
        sys.exit(0)

    salida = (resultado.stdout + resultado.stderr)[-4000:]
    print(f"Los tests rápidos fallan después de editar {ruta}. Corrígelo antes de seguir:\n{salida}",
          file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
