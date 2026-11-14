"""Verifica que el checkpoint guardado sea de verdad el mejor según validación (hallazgo H2).

Espera:
  - metricas.csv con columnas: fase, epoch, val_f1_macro (y otras)
  - un checkpoint {"state_dict", "epoch", "fase", "val_f1_macro", "class_to_idx"}

Con --datos, además recalcula el F1 macro de validación con el checkpoint y lo compara.

Uso:
  python verificacion/verificar_checkpoint.py --metricas training/salidas/metricas.csv \
      --checkpoint training/salidas/mejor.pt [--datos data]
"""
import argparse
import csv
import sys
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from verificacion.referencia import CLASS_TO_IDX  # noqa: E402

CLAVES = {"state_dict", "epoch", "fase", "val_f1_macro", "class_to_idx"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metricas", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--datos")
    args = p.parse_args()
    errores = []

    filas = list(csv.DictReader(open(args.metricas, encoding="utf-8")))
    if not filas:
        sys.exit("metricas.csv está vacío")
    mejor = max(filas, key=lambda f: float(f["val_f1_macro"]))
    fases = {f["fase"] for f in filas}
    print(f"Épocas registradas: {len(filas)} en las fases {sorted(fases)}")
    print(f"Mejor según CSV: fase {mejor['fase']}, época {mejor['epoch']}, F1 val {float(mejor['val_f1_macro']):.4f}")

    ck = torch.load(args.checkpoint, map_location="cpu")
    faltan = CLAVES - set(ck) if isinstance(ck, dict) else CLAVES
    if faltan:
        sys.exit(f"RECHAZADO: el checkpoint no tiene {sorted(faltan)}")
    print(f"Checkpoint: fase {ck['fase']}, época {ck['epoch']}, F1 val {ck['val_f1_macro']:.4f}")

    if str(ck["fase"]) != mejor["fase"] or int(ck["epoch"]) != int(mejor["epoch"]):
        errores.append("el checkpoint no corresponde a la época de mejor F1 en metricas.csv")
    if abs(float(ck["val_f1_macro"]) - float(mejor["val_f1_macro"])) > 1e-6:
        errores.append("el F1 guardado en el checkpoint no coincide con el del CSV")
    if ck["class_to_idx"] != CLASS_TO_IDX:
        errores.append(f"class_to_idx distinto: {ck['class_to_idx']}")

    if args.datos:
        from sklearn.metrics import f1_score
        from torch.utils.data import DataLoader
        from torchvision import datasets
        from verificacion.linea_base import cargar_modelo, preprocesamiento_de_la_app
        modelo = cargar_modelo(args.checkpoint)
        conjunto = datasets.ImageFolder(Path(args.datos) / "valid", transform=preprocesamiento_de_la_app())
        reales, predichas = [], []
        with torch.no_grad():
            for x, y in DataLoader(conjunto, batch_size=32):
                predichas += modelo(x).argmax(1).tolist()
                reales += y.tolist()
        f1 = f1_score(reales, predichas, average="macro")
        print(f"F1 val recalculado con el checkpoint: {f1:.4f}")
        if abs(f1 - float(ck["val_f1_macro"])) > 1e-3:
            errores.append(f"el F1 recalculado ({f1:.4f}) no coincide con el guardado: ¿se guardó otra época?")

    if errores:
        print("\nRECHAZADO:\n- " + "\n- ".join(errores))
        sys.exit(1)
    print("\nAPROBADO: el checkpoint es el mejor modelo registrado.")


if __name__ == "__main__":
    main()
