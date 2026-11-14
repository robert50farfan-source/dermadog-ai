"""Tests del preprocesamiento único (H1) y del recorte que se superpone con Grad-CAM.

Usan imágenes sintéticas generadas aquí con semilla fija, para no depender del dataset.
Las únicas fotos reales opcionales salen de data/valid (según CLAUDE.md, la partición
de evaluación final solo la usa verificacion/linea_base.py).
"""
import logging
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision.transforms as T
from PIL import Image

from backend.app import model_utils
from backend.app.transforms import (
    CROP_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RESIZE_SIZE,
    get_eval_transforms,
    model_crop_box,
)
from verificacion.referencia import transformacion_evaluacion

logger = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parents[1]
# Partición de validación: la única del dataset que pueden usar estos tests (opcional)
CARPETA_VALID = RAIZ / "data" / "valid"

# Proporciones variadas: apaisadas, verticales, cuadradas, muy alargadas, lado corto < 256 y fotos grandes
TAMANOS = [
    (640, 480), (480, 640), (300, 300), (1024, 768), (257, 1000), (1000, 257),
    (333, 517), (256, 256), (225, 400), (199, 151), (4000, 3000), (3000, 4000),
    (3024, 4032), (300, 200), (224, 224),
]


def imagen_ruido(ancho, alto, semilla=0):
    rng = np.random.default_rng(semilla)
    return Image.fromarray(rng.integers(0, 256, size=(alto, ancho, 3), dtype=np.uint8), "RGB")


def imagen_degradado(ancho, alto):
    """Imagen suave cuyo color codifica la posición: si el recorte se desplaza, cambian los píxeles."""
    x = np.linspace(0, 255, ancho)[None, :].repeat(alto, 0)
    y = np.linspace(0, 255, alto)[:, None].repeat(ancho, 1)
    pixeles = np.stack([x, y, (x + y) / 2], axis=-1).astype(np.uint8)
    return Image.fromarray(pixeles, "RGB")


def preprocesamiento_anterior():
    """Copia literal de model_utils.get_transforms() antes de corregir H1 (commit 57b3411)."""
    # Copia FIJA del comportamiento anterior, solo para comparar en estos tests.
    # No debe reutilizarse en ningún otro sitio: la definición única del preprocesamiento
    # está en backend/app/transforms.py (get_eval_transforms).
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def recorte_anterior(image, display_size=448):
    """Copia literal de center_crop_like_model() antes de corregir H1, para medir cuánto se movió."""
    w, h = image.size
    side = min(w, h) * 224 / 256
    left = (w - side) / 2
    top = (h - side) / 2
    return (round(left), round(top), round(left + side), round(top + side))


def imagenes_reales(n=12):
    """Algunas fotos de data/valid si el dataset está disponible (no se sube a Git)."""
    rutas = sorted(CARPETA_VALID.glob("*/*"))
    return [Image.open(p).convert("RGB") for p in rutas[:: max(1, len(rutas) // n)][:n]]


def test_constantes():
    assert (RESIZE_SIZE, CROP_SIZE) == (256, 224)
    assert IMAGENET_MEAN == [0.485, 0.456, 0.406]
    assert IMAGENET_STD == [0.229, 0.224, 0.225]


@pytest.mark.parametrize("ancho,alto", TAMANOS)
def test_igual_a_la_referencia(ancho, alto):
    imagen = imagen_ruido(ancho, alto, semilla=ancho * alto)
    assert torch.equal(get_eval_transforms()(imagen), transformacion_evaluacion()(imagen))


@pytest.mark.parametrize("ancho,alto", TAMANOS)
def test_tensor_identico_al_anterior(ancho, alto):
    """El cambio es solo de organización: el tensor de entrada del modelo no cambia ni un bit."""
    imagen = imagen_ruido(ancho, alto, semilla=ancho + alto)
    assert torch.equal(get_eval_transforms()(imagen), preprocesamiento_anterior()(imagen))


def test_tensor_identico_al_anterior_en_fotos_reales():
    if not CARPETA_VALID.is_dir():
        pytest.skip("data/valid no está disponible")
    fotos = imagenes_reales()
    if not fotos:
        pytest.skip("data/valid no contiene imágenes")
    for foto in fotos:
        assert torch.equal(get_eval_transforms()(foto), preprocesamiento_anterior()(foto))


def test_la_app_usa_get_eval_transforms():
    assert not hasattr(model_utils, "get_transforms")
    assert model_utils.get_eval_transforms is get_eval_transforms


@pytest.mark.parametrize("fabrica", [imagen_ruido, imagen_degradado])
@pytest.mark.parametrize("ancho,alto", TAMANOS)
def test_recorte_grad_cam_coincide_con_torchvision(fabrica, ancho, alto):
    """center_crop_like_model a CROP_SIZE da los mismos píxeles que Resize(256)+CenterCrop(224)."""
    imagen = fabrica(ancho, alto)
    esperado = np.asarray(T.Compose([T.Resize(RESIZE_SIZE), T.CenterCrop(CROP_SIZE)])(imagen), dtype=np.int16)
    obtenido = np.asarray(model_utils.center_crop_like_model(imagen, display_size=CROP_SIZE), dtype=np.int16)
    assert obtenido.shape == esperado.shape
    diferencia = np.abs(obtenido - esperado).max()
    # Medido: 0 o 1 nivel según el tamaño. PIL calcula los coeficientes con el desplazamiento del box
    # en coma flotante y eso puede cambiar el redondeo final de algún píxel; la región es la misma.
    assert diferencia <= 1, f"{ancho}x{alto}: diferencia máxima {diferencia} niveles"


@pytest.mark.parametrize("ancho,alto", TAMANOS)
def test_recorte_grad_cam_es_cuadrado_y_centrado(ancho, alto):
    left, top, right, bottom = model_crop_box(ancho, alto)
    assert 0 <= left < right <= ancho and 0 <= top < bottom <= alto
    # Cuadrado en la imagen original (salvo el truncado de torchvision en el lado largo, < 1 px)
    assert abs((right - left) - (bottom - top)) < max(ancho, alto) / RESIZE_SIZE
    assert abs((right - left) - min(ancho, alto) * CROP_SIZE / RESIZE_SIZE) < max(ancho, alto) / RESIZE_SIZE
    # Centrado salvo el redondeo de CenterCrop (medio píxel de la imagen redimensionada)
    assert abs((left + right) / 2 - ancho / 2) <= ancho / RESIZE_SIZE
    assert abs((top + bottom) / 2 - alto / 2) <= alto / RESIZE_SIZE


@pytest.mark.parametrize("ancho,alto", TAMANOS)
def test_la_region_no_cambia_respecto_al_recorte_anterior(ancho, alto):
    """El recorte nuevo solo corrige redondeos: se mueve menos de un píxel de la imagen redimensionada (+0,5 px)."""
    nuevo = model_crop_box(ancho, alto)
    anterior = recorte_anterior(imagen_ruido(ancho, alto))
    desplazamiento = max(abs(a - b) for a, b in zip(nuevo, anterior))
    detalle = (f"{ancho}x{alto}: caja nueva {tuple(round(v, 2) for v in nuevo)} "
               f"anterior {anterior} desplazamiento máx {desplazamiento:.2f} px")
    logger.info(detalle)
    assert desplazamiento <= max(ancho, alto) / RESIZE_SIZE + 0.5, detalle
