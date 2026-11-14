#!/usr/bin/env python3
"""Orquestador de DermaDog: ejecuta fases con agentes de Claude Code y controles automáticos.

Cada fase:
  1. crea una rama agente/<fase> desde la rama actual;
  2. le da la tarea a un subagente con `claude -p` (modo sin interfaz);
  3. corre los controles (tests, métricas, revisor);
  4. si fallan, le devuelve el error al agente (máx. 3 intentos);
  5. si pasan, hace commit en la rama. Nunca hace push ni despliega: eso lo decide el humano.

Uso:
  python orquestador.py plan                 # muestra las fases y sus controles
  python orquestador.py ejecutar F1          # ejecuta una fase
  python orquestador.py ejecutar F1 --simular   # muestra qué haría, sin llamar a Claude
  python orquestador.py controles F1         # solo corre los controles de la fase
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
PY = sys.executable
MAX_INTENTOS = 3
PYTEST_RAPIDO = f'{PY} -m pytest verificacion/tests -m "not hallazgo and not lento" -q -p no:cacheprovider'
# En Windows, subprocess decodifica con cp1252 y se cae con texto UTF-8 (por ejemplo "Á" o emojis).
# Se lee todo como UTF-8 y se pide lo mismo a los procesos Python hijos.
ENTORNO = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
TEXTO = {"text": True, "encoding": "utf-8", "errors": "replace", "env": ENTORNO}


def hay_datos():
    return (RAIZ / "data/test").is_dir() and (RAIZ / "backend/models/model_final_resnet50.pth").stat().st_size > 1_000_000


def hay_linea_base():
    return (RAIZ / "verificacion/linea_base.json").exists()


FASES = {
    "F0": {
        "nombre": "Línea base (la ejecuta el humano)",
        "agente": None,
        "requisito": "data/ descargado de Kaggle y modelo real (git lfs pull)",
        "controles": [
            f"{PY} verificacion/linea_base.py --datos data --preprocesamiento app --salida verificacion/linea_base.json",
            f"{PY} verificacion/linea_base.py --datos data --preprocesamiento entrenamiento "
            f"--salida verificacion/resultados/linea_base_entrenamiento.json",
        ],
    },
    "F1": {
        "nombre": "H1: un solo preprocesamiento para entrenamiento y app",
        "agente": "dev-api",
        "tarea": (
            "Corrige el hallazgo H1 de CLAUDE.md. Crea backend/app/transforms.py con get_eval_transforms() "
            "idéntica a la del entrenamiento (Resize((224, 224)) + ToTensor + Normalize de ImageNet) y úsala en "
            "model_utils.py. Ajusta center_crop_like_model para que la imagen que se superpone con Grad-CAM sea "
            "la misma región que ve el modelo (la imagen completa redimensionada). Actualiza README.md si cambia algo visible."
        ),
        "controles": [
            f'{PY} -m pytest verificacion/tests -m hallazgo -k H1 -q -p no:cacheprovider',
            PYTEST_RAPIDO,
            "METRICAS_APP",
            "REVISOR",
        ],
    },
    "F2": {
        "nombre": "H3, H5, H6, H8 y H4: robustez de la API",
        "agente": "dev-api",
        "tarea": (
            "Corrige los hallazgos H3, H4, H5, H6 y H8 de CLAUDE.md: carga estricta del modelo (si falla, la app "
            "no debe servir predicciones), un threading.Lock alrededor de la predicción y Grad-CAM, "
            "register_full_backward_hook, límite de 10 MB en /predict con respuesta 413, y CORS sin comodín con "
            "credenciales. No cambies umbrales ni el contrato de /predict."
        ),
        "controles": [
            f'{PY} -m pytest verificacion/tests -m hallazgo -k "H3 or H5 or H6 or H8" -q -p no:cacheprovider',
            PYTEST_RAPIDO,
            "REVISOR",
        ],
    },
    "F3": {
        "nombre": "H2: entrenamiento reproducible que guarda el mejor modelo",
        "agente": "entrenador",
        "tarea": (
            "Convierte notebooks/torchDogF.ipynb en training/entrenar.py (ejecutable en Kaggle con GPU) cumpliendo "
            "todas las reglas de tu definición: transformaciones importadas de backend/app/transforms.py, "
            "eliminación de duplicados, F1 macro de validación en ambas fases, copy.deepcopy del mejor estado, "
            "checkpoint con metadatos en training/salidas/mejor.pt y metricas.csv. Agrega training/README.md "
            "con cómo ejecutarlo en Kaggle. No ejecutes el entrenamiento aquí."
        ),
        "controles": [
            f"{PY} -m py_compile training/entrenar.py",
            PYTEST_RAPIDO,
            "REVISOR",
        ],
    },
    "F4": {
        "nombre": "Evaluar el modelo reentrenado (después de correr F3 en Kaggle)",
        "agente": None,
        "requisito": "training/salidas/mejor.pt y metricas.csv descargados de Kaggle",
        "controles": [
            f"{PY} verificacion/verificar_checkpoint.py --metricas training/salidas/metricas.csv "
            f"--checkpoint training/salidas/mejor.pt --datos data",
            f"{PY} verificacion/linea_base.py --datos data --modelo training/salidas/mejor.pt "
            f"--salida verificacion/resultados/candidato.json",
            f"{PY} verificacion/comparar.py verificacion/linea_base.json verificacion/resultados/candidato.json",
        ],
    },
}


def correr(comando, simular=False):
    print(f"$ {comando}")
    if simular:
        return 0, ""
    r = subprocess.run(comando, shell=True, cwd=RAIZ, capture_output=True, **TEXTO)
    salida = (r.stdout + r.stderr).strip()
    print(salida[-3000:])
    return r.returncode, salida


def claude(prompt, simular=False):
    comando = ["claude", "-p", prompt, "--output-format", "json",
               "--permission-mode", "acceptEdits", "--max-turns", "60"]
    print(f"$ claude -p {shlex.quote(prompt[:90])}...")
    if simular:
        return ""
    r = subprocess.run(comando, cwd=RAIZ, capture_output=True, **TEXTO)
    try:
        return json.loads(r.stdout).get("result", "")
    except json.JSONDecodeError:
        return r.stdout + r.stderr


def control_metricas_app(simular):
    """Mide la app con el código actual y la compara con la línea base."""
    if not (hay_datos() and hay_linea_base()):
        print("(sin data/ o sin línea base: se omite el control de métricas)")
        return 0, ""
    codigo, salida = correr(f"{PY} verificacion/linea_base.py --datos data --preprocesamiento app "
                            f"--salida verificacion/resultados/candidato_app.json", simular)
    if codigo:
        return codigo, salida
    return correr(f"{PY} verificacion/comparar.py verificacion/linea_base.json "
                  f"verificacion/resultados/candidato_app.json", simular)


def control_revisor(simular):
    respuesta = claude("Usa el subagente revisor para revisar los cambios de esta rama: `git status` y `git diff HEAD`, "
                       "incluidos los archivos nuevos sin seguimiento. "
                       "Devuelve su informe completo.", simular)
    print(respuesta[-3000:])
    if simular or "VEREDICTO: APROBADO" in respuesta:
        return 0, respuesta
    return 1, respuesta


def correr_controles(fase, simular=False):
    for control in FASES[fase]["controles"]:
        if control == "METRICAS_APP":
            codigo, salida = control_metricas_app(simular)
        elif control == "REVISOR":
            codigo, salida = control_revisor(simular)
        else:
            codigo, salida = correr(control, simular)
        if codigo:
            return False, f"Falló el control `{control}`:\n{salida[-3000:]}"
    return True, ""


def ejecutar(fase, simular=False):
    datos = FASES[fase]
    print(f"\n=== {fase}: {datos['nombre']} ===")
    if datos["agente"] is None:
        print(f"Fase manual. Requisito: {datos['requisito']}")
        ok, error = correr_controles(fase, simular)
        print("CONTROLES OK" if ok else error)
        return ok

    rama = f"agente/{fase.lower()}"
    correr(f"git switch -c {rama} || git switch {rama}", simular)
    prompt = f"Usa el subagente {datos['agente']}. Tarea: {datos['tarea']}"
    for intento in range(1, MAX_INTENTOS + 1):
        print(f"\n--- intento {intento}/{MAX_INTENTOS} ---")
        print(claude(prompt, simular)[-2000:])
        ok, error = correr_controles(fase, simular)
        if ok:
            correr(f'git add -A && git commit -m "{fase}: {datos["nombre"]}"', simular)
            print(f"\n{fase} APROBADA en la rama {rama}. Revise el diff y decida si la fusiona.")
            return True
        prompt = (f"Usa el subagente {datos['agente']}. Los controles de la fase {fase} fallaron. "
                  f"Corrige el problema sin modificar verificacion/ ni backend/models/.\n{error}")
    print(f"\n{fase} NO pasó tras {MAX_INTENTOS} intentos. Queda en la rama {rama} para revisión humana.")
    return False


def main():
    # Si la salida va a un archivo o a una tubería, un carácter no representable no detiene el orquestador
    sys.stdout.reconfigure(errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("accion", choices=["plan", "ejecutar", "controles"])
    p.add_argument("fase", nargs="?", choices=list(FASES))
    p.add_argument("--simular", action="store_true")
    args = p.parse_args()

    if args.accion == "plan":
        for clave, datos in FASES.items():
            quien = f"agente {datos['agente']}" if datos["agente"] else "humano"
            print(f"{clave}  {datos['nombre']}  [{quien}]")
            for c in datos["controles"]:
                print(f"      control: {c}")
        return
    if not args.fase:
        p.error("indique la fase, por ejemplo: ejecutar F1")
    if args.accion == "controles":
        ok, error = correr_controles(args.fase, args.simular)
        print("CONTROLES OK" if ok else error)
        sys.exit(0 if ok else 1)
    sys.exit(0 if ejecutar(args.fase, args.simular) else 1)


if __name__ == "__main__":
    main()
