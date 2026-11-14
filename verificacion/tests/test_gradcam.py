"""Grad-CAM: forma, rango y que explique la clase pedida."""
import warnings

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def modelo_y_cam():
    from backend.app import model_utils
    from backend.app.gradcam import GradCAM
    torch.manual_seed(0)
    modelo = model_utils.load_model(path="no-existe.pth", num_classes=6)
    return modelo, GradCAM(modelo, modelo.layer4)


def test_mapa_de_calor_tiene_forma_y_rango_validos(modelo_y_cam):
    _, cam = modelo_y_cam
    mapa = cam.generate_heatmap(torch.randn(1, 3, 224, 224), 0)
    assert mapa.shape == (7, 7)
    assert np.isfinite(mapa).all() and mapa.min() >= 0 and mapa.max() <= 1


def test_mapa_depende_de_la_clase_explicada(modelo_y_cam):
    _, cam = modelo_y_cam
    torch.manual_seed(1)
    x = torch.randn(1, 3, 224, 224)
    mapas = [cam.generate_heatmap(x, clase) for clase in range(6)]
    assert any(not np.allclose(mapas[0], m) for m in mapas[1:])


@pytest.mark.hallazgo("H5")
def test_H5_no_usa_hooks_de_retropropagacion_obsoletos(modelo_y_cam):
    _, cam = modelo_y_cam
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        cam.generate_heatmap(torch.randn(1, 3, 224, 224), 2)
    obsoletos = [str(a.message) for a in avisos if "backward hook" in str(a.message).lower()]
    assert not obsoletos, obsoletos[0]
