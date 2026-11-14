"""Compara un resultado candidato con la línea base. Sale con código 1 si empeora.

Reglas (ajustables solo por el humano):
  - F1 macro no puede bajar más de 0,01.
  - El recall de ninguna clase puede bajar más de 0,05.
  - Las tiñas clasificadas como "Healthy" no pueden aumentar.
  - Si se pasan resultados del set de oro, la tasa de aciertos no puede bajar.

Uso:
  python verificacion/comparar.py verificacion/linea_base.json verificacion/resultados/candidato.json \
      [--oro-base verificacion/resultados/oro_base.json --oro-candidato verificacion/resultados/oro_candidato.json]
"""
import argparse
import json
import sys
from pathlib import Path

TOLERANCIA_F1 = 0.01
TOLERANCIA_RECALL = 0.05


def leer(ruta):
    return json.loads(Path(ruta).read_text())


def tinas_como_sanas(resultado):
    clases = resultado["clases"]
    return resultado["matriz_confusion"][clases.index("ringworm")][clases.index("Healthy")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("base")
    p.add_argument("candidato")
    p.add_argument("--oro-base")
    p.add_argument("--oro-candidato")
    args = p.parse_args()

    base, cand = leer(args.base), leer(args.candidato)
    problemas = []

    print(f"{'Métrica':<28}{'Base':>10}{'Candidato':>12}")
    print(f"{'F1 macro':<28}{base['f1_macro']:>10.4f}{cand['f1_macro']:>12.4f}")
    print(f"{'Exactitud':<28}{base['exactitud']:>10.4f}{cand['exactitud']:>12.4f}")
    if cand["f1_macro"] < base["f1_macro"] - TOLERANCIA_F1:
        problemas.append(f"F1 macro baja de {base['f1_macro']:.4f} a {cand['f1_macro']:.4f}")

    for clase, valores in base["por_clase"].items():
        r_base, r_cand = valores["recall"], cand["por_clase"][clase]["recall"]
        print(f"{'Recall ' + clase:<28}{r_base:>10.4f}{r_cand:>12.4f}")
        if r_cand < r_base - TOLERANCIA_RECALL:
            problemas.append(f"recall de {clase} baja de {r_base:.4f} a {r_cand:.4f}")

    t_base, t_cand = tinas_como_sanas(base), tinas_como_sanas(cand)
    print(f"{'Tiña predicha como sana':<28}{t_base:>10}{t_cand:>12}")
    if t_cand > t_base:
        problemas.append(f"tiñas clasificadas como sanas suben de {t_base} a {t_cand}")

    if args.oro_base and args.oro_candidato:
        a_base, a_cand = leer(args.oro_base)["resumen"]["aciertos"], leer(args.oro_candidato)["resumen"]["aciertos"]
        print(f"{'Aciertos set de oro':<28}{a_base:>10}{a_cand:>12}")
        if a_cand < a_base:
            problemas.append(f"aciertos del set de oro bajan de {a_base} a {a_cand}")

    if base["n"] != cand["n"] or base["particion"] != cand["particion"]:
        problemas.append("los resultados no se midieron sobre el mismo conjunto")

    if problemas:
        print("\nRECHAZADO:\n- " + "\n- ".join(problemas))
        sys.exit(1)
    print("\nAPROBADO: el candidato no empeora la línea base.")


if __name__ == "__main__":
    main()
