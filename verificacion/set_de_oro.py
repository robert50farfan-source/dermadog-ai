"""Pasa las fotos externas del set de oro por la app completa (nitidez, CLIP, modelo, umbral).

manifest.csv (en verificacion/set_de_oro/):
  archivo,clase_esperada,fuente
  tina_01.jpg,ringworm,foto propia
  sarna_01.jpg,fuera_de_clase,internet (url)
  gato_01.jpg,no_perro,foto propia

clase_esperada es una de las 6 clases, "fuera_de_clase" (enfermedad que el modelo no conoce)
o "no_perro". Para fuera_de_clase y no_perro, lo correcto es que la app NO dé un diagnóstico.

Uso:
  python verificacion/set_de_oro.py --salida verificacion/resultados/oro_actual.json
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from backend.app import model_utils  # noqa: E402

CARPETA = RAIZ / "verificacion/set_de_oro"
SIN_DIAGNOSTICO = {"fuera_de_clase", "no_perro"}


def es_acierto(esperada, resultado):
    if esperada in SIN_DIAGNOSTICO:
        return resultado["status"] != "ok"
    return resultado["status"] == "ok" and resultado["class_key"] == esperada


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=str(CARPETA / "manifest.csv"))
    p.add_argument("--salida", required=True)
    args = p.parse_args()

    filas = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    if not filas:
        sys.exit("El manifest está vacío: agregue fotos con su clase confirmada.")
    model_utils.initialize()

    detalle = []
    for fila in filas:
        with open(CARPETA / fila["archivo"], "rb") as f:
            r = model_utils.predict_image(f)
        detalle.append({
            "archivo": fila["archivo"],
            "esperada": fila["clase_esperada"],
            "status": r["status"],
            "predicha": r["class_key"],
            "confianza": r["confidence"],
            "acierto": es_acierto(fila["clase_esperada"], r),
        })

    resumen = {
        "total": len(detalle),
        "aciertos": sum(d["acierto"] for d in detalle),
        "por_status": dict(Counter(d["status"] for d in detalle)),
        "tinas_como_sanas": sum(d["esperada"] == "ringworm" and d["predicha"] == "Healthy" for d in detalle),
    }
    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    Path(args.salida).write_text(json.dumps({"resumen": resumen, "detalle": detalle}, indent=2, ensure_ascii=False))

    print(f"{'Archivo':<24}{'Esperada':<18}{'Status':<16}{'Predicha':<20}{'Conf.':>6}  OK")
    for d in detalle:
        conf = f"{d['confianza']:.2f}" if d["confianza"] is not None else "-"
        print(f"{d['archivo']:<24}{d['esperada']:<18}{d['status']:<16}{str(d['predicha']):<20}{conf:>6}  "
              f"{'sí' if d['acierto'] else 'NO'}")
    print(f"\nAciertos: {resumen['aciertos']}/{resumen['total']} · tiñas como sanas: {resumen['tinas_como_sanas']}")


if __name__ == "__main__":
    main()
