"""Configuración común de los tests de verificación de DermaDog."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
# main.py sirve frontend/ con una ruta relativa: los tests corren desde la raíz
os.chdir(RAIZ)


def pytest_configure(config):
    config.addinivalue_line("markers", "hallazgo(id): falla hasta que se corrija el hallazgo de la auditoría")
    config.addinivalue_line("markers", "lento: necesita pesos reales, dataset o descargas")


def imagen_con_textura(ancho, alto, semilla=0):
    """Imagen RGB con ruido: tiene nitidez alta y pasa el filtro de borrosidad."""
    rng = np.random.default_rng(semilla)
    pixeles = rng.integers(0, 256, size=(alto, ancho, 3), dtype=np.uint8)
    return Image.fromarray(pixeles, "RGB")


@pytest.fixture
def model_utils_aislado(monkeypatch):
    """model_utils sin cargar ResNet ni CLIP de verdad (no necesita pesos ni internet)."""
    from backend.app import model_utils
    from verificacion.referencia import CLASS_TO_IDX

    monkeypatch.setattr(model_utils, "initialize", lambda: None)
    monkeypatch.setattr(model_utils, "model", None)
    monkeypatch.setattr(model_utils, "grad_cam", None)
    monkeypatch.setattr(model_utils, "clip_model", None)
    monkeypatch.setattr(model_utils, "clip_processor", None)
    monkeypatch.setattr(model_utils, "idx_to_class", {v: k for k, v in CLASS_TO_IDX.items()})
    return model_utils
