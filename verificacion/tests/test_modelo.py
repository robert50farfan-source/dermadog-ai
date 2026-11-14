"""Carga del modelo y de las clases."""
import json
from pathlib import Path

import pytest
import torch

from verificacion.referencia import CLASS_TO_IDX

RUTA_MODELO = Path("backend/models/model_final_resnet50.pth")


def es_puntero_lfs(ruta):
    return ruta.exists() and ruta.stat().st_size < 1_000_000


def test_clases_coinciden_con_el_entrenamiento():
    assert json.loads(Path("backend/models/class_to_idx.json").read_text()) == CLASS_TO_IDX


@pytest.mark.lento
def test_pesos_reales_cargan_en_modo_estricto():
    if es_puntero_lfs(RUTA_MODELO):
        pytest.skip("El modelo es un puntero de Git LFS: ejecute `git lfs pull`.")
    from backend.app import model_utils
    estado = torch.load(RUTA_MODELO, map_location="cpu")
    if isinstance(estado, dict) and "state_dict" in estado:
        estado = estado["state_dict"]
    modelo = model_utils.load_model(path="no-existe.pth", num_classes=6)
    modelo.load_state_dict(estado, strict=True)
    assert modelo.fc[-1].out_features == 6


@pytest.mark.hallazgo("H3")
def test_H3_pesos_incompletos_detienen_la_carga(tmp_path):
    """Si faltan pesos, la app no debe arrancar con un modelo a medio cargar."""
    from backend.app import model_utils
    modelo = model_utils.load_model(path="no-existe.pth", num_classes=6)
    estado = {k: v for k, v in modelo.state_dict().items() if not k.startswith("layer4")}
    ruta = tmp_path / "incompleto.pth"
    torch.save(estado, ruta)
    with pytest.raises(Exception):
        model_utils.load_model(path=str(ruta), num_classes=6)
