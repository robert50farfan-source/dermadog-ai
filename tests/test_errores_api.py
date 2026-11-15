"""Errores de /predict: códigos correctos y mensajes en español, sin detalles internos.

No necesitan pesos reales, CLIP ni el dataset.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import model_utils

RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture
def cliente(monkeypatch):
    """La app con un modelo ficticio ya "cargado": predict_image llega a abrir la imagen."""
    # main.py monta frontend/ con ruta relativa
    monkeypatch.chdir(RAIZ)
    monkeypatch.setattr(model_utils, "initialize", lambda: None)
    monkeypatch.setattr(model_utils, "model", object())
    monkeypatch.setattr(model_utils, "load_error", None)
    from backend.app import main
    with TestClient(main.app) as c:
        yield c


def test_archivo_que_no_es_imagen_da_400_en_espanol(cliente):
    r = cliente.post("/predict", files={"file": ("nota.txt", b"hola", "text/plain")})
    assert r.status_code == 400
    assert r.json()["detail"] == "El archivo debe ser una imagen."


def test_imagen_danada_da_400_y_no_500(cliente):
    """Content-Type de imagen pero contenido ilegible: es un error del archivo, no del servidor."""
    r = cliente.post("/predict", files={"file": ("rota.jpg", b"esto no es un jpeg", "image/jpeg")})
    assert r.status_code == 400
    assert r.json()["detail"] == "El archivo no es una imagen válida o está dañado."


def test_imagen_con_demasiados_pixeles_da_400(cliente, monkeypatch):
    def bomba(stream):
        raise Image.DecompressionBombError("demasiados píxeles")

    monkeypatch.setattr(model_utils, "predict_image", bomba)
    r = cliente.post("/predict", files={"file": ("enorme.jpg", b"\xff\xd8\xff", "image/jpeg")})
    assert r.status_code == 400
    assert "píxeles" in r.json()["detail"]


def test_error_inesperado_da_500_sin_mostrar_detalles_internos(cliente, monkeypatch):
    def falla(stream):
        raise RuntimeError("detalle interno secreto en /ruta/del/servidor")

    monkeypatch.setattr(model_utils, "predict_image", falla)
    r = cliente.post("/predict", files={"file": ("p.jpg", b"\xff\xd8\xff", "image/jpeg")})
    assert r.status_code == 500
    detalle = r.json()["detail"]
    assert detalle == "Ocurrió un error inesperado al procesar la imagen."
    assert "secreto" not in detalle and "/ruta" not in detalle
