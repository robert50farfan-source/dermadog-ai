"""Mide el clasificador de DermaDog en una partición del dataset y guarda las métricas en JSON.

Evalúa solo el clasificador (sin filtros de nitidez ni CLIP), con el preprocesamiento elegido:
  --preprocesamiento app            -> el que usa hoy la app (model_utils)
  --preprocesamiento entrenamiento  -> la definición única (verificacion/referencia.py)

Ejemplos:
  python verificacion/linea_base.py --datos data --salida verificacion/linea_base.json
  python verificacion/linea_base.py --datos data --modelo training/salidas/mejor.pt \
      --salida verificacion/resultados/candidato.json
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from backend.app import model_utils  # noqa: E402
from verificacion.referencia import CLASS_TO_IDX, transformacion_evaluacion  # noqa: E402


def preprocesamiento_de_la_app():
    """Lo que la app le aplica a la imagen antes del modelo."""
    try:  # después de corregir H1 debería existir este módulo compartido
        from backend.app.transforms import get_eval_transforms
        return get_eval_transforms()
    except ImportError:
        return model_utils.get_transforms()


def cargar_modelo(ruta):
    """Acepta un state_dict suelto o un checkpoint {"state_dict": ..., ...}."""
    contenido = torch.load(ruta, map_location="cpu")
    estado = contenido["state_dict"] if isinstance(contenido, dict) and "state_dict" in contenido else contenido
    modelo = model_utils.load_model(path="no-existe.pth", num_classes=len(CLASS_TO_IDX))
    modelo.load_state_dict(estado, strict=True)
    return modelo.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datos", required=True, help="carpeta con train/ valid/ test/")
    p.add_argument("--particion", default="test", choices=["test", "valid"])
    p.add_argument("--modelo", default="backend/models/model_final_resnet50.pth")
    p.add_argument("--preprocesamiento", default="app", choices=["app", "entrenamiento"])
    p.add_argument("--salida", required=True)
    args = p.parse_args()

    ruta_modelo = Path(args.modelo)
    if ruta_modelo.stat().st_size < 1_000_000:
        sys.exit(f"{ruta_modelo} es un puntero de Git LFS. Ejecute `git lfs pull` primero.")

    transformacion = preprocesamiento_de_la_app() if args.preprocesamiento == "app" else transformacion_evaluacion()
    conjunto = datasets.ImageFolder(Path(args.datos) / args.particion, transform=transformacion)
    if conjunto.class_to_idx != CLASS_TO_IDX:
        sys.exit(f"Las clases del dataset {conjunto.class_to_idx} no coinciden con las del modelo {CLASS_TO_IDX}")

    modelo = cargar_modelo(ruta_modelo)
    reales, predichas = [], []
    with torch.no_grad():
        for x, y in DataLoader(conjunto, batch_size=32, shuffle=False, num_workers=2):
            predichas += modelo(x).argmax(1).tolist()
            reales += y.tolist()

    nombres = list(CLASS_TO_IDX)
    reporte = classification_report(reales, predichas, target_names=nombres, output_dict=True, zero_division=0)
    resultado = {
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modelo": str(ruta_modelo),
        "particion": args.particion,
        "preprocesamiento": args.preprocesamiento,
        "n": len(reales),
        "exactitud": reporte["accuracy"],
        "f1_macro": f1_score(reales, predichas, average="macro"),
        "por_clase": {c: {k: reporte[c][k] for k in ("precision", "recall", "f1-score", "support")} for c in nombres},
        "clases": nombres,
        "matriz_confusion": confusion_matrix(reales, predichas, labels=list(range(len(nombres)))).tolist(),
    }
    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    Path(args.salida).write_text(json.dumps(resultado, indent=2, ensure_ascii=False))
    print(classification_report(reales, predichas, target_names=nombres, digits=4, zero_division=0))
    print(f"F1 macro: {resultado['f1_macro']:.4f} · guardado en {args.salida}")


if __name__ == "__main__":
    main()
