# Kit de agentes para DermaDog AI

Automatiza la mejora de DermaDog con subagentes de Claude Code, hooks y controles automáticos.
La idea central: **los agentes escriben código, pero quien decide si un cambio pasa es código de verificación que ellos no pueden modificar.**

## Contenido

| Ruta | Qué es | ¿Lo puede modificar un agente? |
| --- | --- | --- |
| `CLAUDE.md` | Contexto, reglas y hallazgos abiertos | Sí, con cuidado |
| `.claude/agents/` | 5 subagentes: revisor, evaluador, entrenador, dev-api, analista-datos | No debería |
| `.claude/settings.json` | Permisos y hooks | No (bloqueado) |
| `.claude/hooks/guardia.py` | Bloquea push, despliegue, borrados, tocar `verificacion/`, `backend/models/` y `data/test` | No (bloqueado) |
| `.claude/hooks/post_edicion.py` | Tras editar un `.py`, corre los tests rápidos; si fallan, el agente debe corregir | No (bloqueado) |
| `verificacion/tests/` | Tests de contrato y de hallazgos (`-m hallazgo`) | No (bloqueado) |
| `verificacion/linea_base.py` | Métricas del modelo en test (sklearn) → JSON | No (bloqueado) |
| `verificacion/comparar.py` | Rechaza un candidato si empeora la línea base | No (bloqueado) |
| `verificacion/verificar_checkpoint.py` | Comprueba que el modelo guardado sea el mejor (H2) | No (bloqueado) |
| `verificacion/set_de_oro.py` | Pasa fotos externas por la app completa | No (bloqueado) |
| `verificacion/set_de_oro/` | Fotos externas y `manifest.csv` | Sí |
| `orquestador.py` | Ejecuta fases: agente → controles → reintento → commit | No (bloqueado) |

## Instalación

1. Copie el contenido del kit en la raíz del repositorio `dermadog-ai`.
2. Copie `torchDogF.ipynb` en `notebooks/`.
3. Agregue al `.gitignore`: `data/`, `verificacion/resultados/*.json`, `training/salidas/`.
4. Cree el entorno en `.venv/` o `venv/` dentro del repositorio (el hook de tests lo busca ahí) e instale:
   `pip install -r backend/requirements.txt pytest httpx scikit-learn`.
5. Descargue los pesos reales: `git lfs pull`.
6. Pruebe: `python -m pytest verificacion/tests -m "not hallazgo" -q` (debe pasar) y
   `python -m pytest verificacion/tests -m hallazgo -q` (hoy falla: son los hallazgos abiertos).

## Flujo

```
python orquestador.py plan
python orquestador.py ejecutar F0     # humano: línea base (requiere data/ y modelo real)
python orquestador.py ejecutar F1     # agente dev-api corrige H1 en la rama agente/f1
python orquestador.py ejecutar F2     # agente dev-api corrige H3-H8 en la rama agente/f2
python orquestador.py ejecutar F3     # agente entrenador escribe training/entrenar.py
#   → usted lo corre en Kaggle con GPU y descarga mejor.pt y metricas.csv
python orquestador.py ejecutar F4     # controles sobre el modelo nuevo
```

Use `--simular` para ver qué haría cada fase sin llamar a Claude.

## Decisiones que quedan en manos humanas

1. Descargar el dataset y aceptar su licencia (F0).
2. Armar el set de oro con fotos de clase confirmada.
3. Fusionar cada rama `agente/*` después de leer el diff.
4. Reemplazar `backend/models/model_final_resnet50.pth` por el modelo nuevo (después de F4).
5. Desplegar en Dokploy.
