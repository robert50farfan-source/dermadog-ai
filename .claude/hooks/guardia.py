#!/usr/bin/env python3
"""Hook PreToolUse: bloquea acciones que un agente no debe hacer en DermaDog.

Claude Code envía por stdin un JSON con tool_name y tool_input.
Salida 0 = permitir. Salida 2 = bloquear; el texto en stderr le llega al agente.
"""
import json
import re
import sys

# Rutas que solo modifica el humano: la vara de medir y el modelo en producción
PROTEGIDAS = (
    "backend/models/",
    "verificacion/",
    ".claude/hooks/",
    ".claude/settings.json",
    "orquestador.py",
)
# Dentro de verificacion/, los agentes sí pueden escribir aquí
EXCEPCIONES = (
    "verificacion/set_de_oro/",
    "verificacion/resultados/",
)

# Comandos que publican, despliegan o destruyen
COMANDOS_PROHIBIDOS = [
    (r"\bgit\s+push\b", "push al remoto"),
    (r"\bdocker\s+push\b", "publicar imagen Docker"),
    (r"dokploy", "despliegue en Dokploy"),
    (r"\brm\s+-[a-z]*r[a-z]*f|\brm\s+-[a-z]*f[a-z]*r", "borrado recursivo"),
    (r"\b(Remove-Item|ri|rd|rmdir|del|erase)\b.*\s-r(ecurse)?\b", "borrado recursivo"),
    (r"\bgit\s+reset\s+--hard\b", "descartar cambios"),
    (r"\bkaggle\s+datasets\s+(create|version)\b", "publicar datos en Kaggle"),
]


def normalizar(ruta):
    return ruta.replace("\\", "/")


def es_protegida(ruta):
    # Sin comillas y con "/" final, para detectar también la carpeta sola (rm -r verificacion)
    ruta = normalizar(ruta).strip("'\"").rstrip("/") + "/"
    if any(e in ruta for e in EXCEPCIONES):
        return False
    return any(p in ruta for p in PROTEGIDAS)


def destinos_de_escritura(comando):
    """Rutas que un comando de shell podría escribir o borrar (heurística)."""
    destinos = re.findall(r">>?\s*([^\s;&|]+)", comando)
    for segmento in re.split(r"&&|\|\||[;|]", comando):
        partes = segmento.split()
        if not partes:
            continue
        programa = partes[0]
        if programa.lower() in ESCRITURA or (programa == "sed" and "-i" in partes):
            destinos += [p for p in partes[1:] if not p.startswith("-")]
    return destinos


# Programas de shell y cmdlets de PowerShell (con sus alias) que escriben, mueven o borran archivos
ESCRITURA = {
    "tee", "mv", "cp", "rm", "touch", "truncate",
    "set-content", "sc", "add-content", "ac", "out-file", "clear-content", "clc", "tee-object",
    "remove-item", "ri", "del", "erase", "rd", "rmdir", "move-item", "mi", "move",
    "copy-item", "cpi", "copy", "new-item", "ni", "rename-item", "rni",
}


def bloquear(motivo):
    print(f"BLOQUEADO por guardia.py: {motivo}. Si es necesario, pídeselo al humano.", file=sys.stderr)
    sys.exit(2)


def main():
    try:
        evento = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)

    herramienta = evento.get("tool_name", "")
    entrada = evento.get("tool_input", {}) or {}

    if herramienta in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        ruta = entrada.get("file_path") or entrada.get("notebook_path") or ""
        if es_protegida(ruta):
            bloquear(f"{ruta} es de solo lectura para los agentes")

    # La herramienta PowerShell (Windows) recibe los mismos controles que Bash
    if herramienta in ("Bash", "PowerShell"):
        comando = entrada.get("command", "")
        for patron, motivo in COMANDOS_PROHIBIDOS:
            if re.search(patron, comando, re.IGNORECASE):
                bloquear(motivo)
        # Escritura por shell sobre rutas protegidas (redirección, sed -i, mv, cp, tee)
        for destino in destinos_de_escritura(comando):
            if es_protegida(destino):
                bloquear(f"escritura por shell sobre la ruta protegida {destino}")
        # data/test solo se toca desde la evaluación oficial
        if "data/test" in comando and "verificacion/linea_base.py" not in comando:
            bloquear("data/test solo se usa a través de verificacion/linea_base.py")

    sys.exit(0)


if __name__ == "__main__":
    main()
