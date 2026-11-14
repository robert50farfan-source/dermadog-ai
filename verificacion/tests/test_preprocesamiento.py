"""El tensor que recibe el modelo en la app debe ser el mismo que vio al entrenar."""
import io

import numpy as np
import pytest
import torch

from verificacion.conftest import imagen_con_textura
from verificacion.referencia import transformacion_evaluacion


class ModeloQueCaptura(torch.nn.Module):
    """Sustituto del ResNet: guarda el tensor de entrada y responde siempre clase 0."""
    def __init__(self):
        super().__init__()
        self.entrada = None

    def forward(self, x):
        self.entrada = x.detach().clone()
        logits = torch.full((x.shape[0], 6), -10.0)
        logits[:, 0] = 10.0
        return logits


class GradCamNulo:
    def generate_heatmap(self, tensor, clase):
        return np.zeros((7, 7), dtype=np.float32)


def tensor_que_ve_el_modelo(model_utils, imagen):
    capturador = ModeloQueCaptura()
    model_utils.model = capturador
    model_utils.grad_cam = GradCamNulo()
    buffer = io.BytesIO()
    imagen.save(buffer, format="PNG")
    buffer.seek(0)
    resultado = model_utils.predict_image(buffer)
    assert resultado["status"] == "ok", resultado
    return capturador.entrada


def test_el_modelo_recibe_un_tensor_de_224(model_utils_aislado):
    x = tensor_que_ve_el_modelo(model_utils_aislado, imagen_con_textura(640, 480))
    assert tuple(x.shape) == (1, 3, 224, 224)


def test_acepta_imagenes_con_canal_alfa(model_utils_aislado):
    imagen = imagen_con_textura(300, 300).convert("RGBA")
    x = tensor_que_ve_el_modelo(model_utils_aislado, imagen)
    assert x.shape[1] == 3


@pytest.mark.hallazgo("H1")
@pytest.mark.parametrize("ancho,alto", [(640, 480), (480, 640), (300, 300), (1024, 768)])
def test_H1_app_y_entrenamiento_usan_el_mismo_preprocesamiento(model_utils_aislado, ancho, alto):
    imagen = imagen_con_textura(ancho, alto, semilla=ancho + alto)
    en_la_app = tensor_que_ve_el_modelo(model_utils_aislado, imagen)
    en_el_entrenamiento = transformacion_evaluacion()(imagen).unsqueeze(0)
    diferencia = (en_la_app - en_el_entrenamiento).abs().max().item()
    assert diferencia < 1e-4, f"diferencia máxima {diferencia:.3f} para {ancho}x{alto}"
