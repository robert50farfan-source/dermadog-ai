"""Contrato de /predict y del servidor (no necesita pesos ni CLIP)."""
import io

import pytest
from fastapi.testclient import TestClient

from verificacion.conftest import imagen_con_textura

CAMPOS = {"status", "message", "prediction", "class_key", "confidence", "top",
          "analyzed_image", "gradcam_image"}


def jpeg(ancho=320, alto=240):
    buffer = io.BytesIO()
    imagen_con_textura(ancho, alto).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def cliente(model_utils_aislado, monkeypatch):
    def predict_falso(stream):
        imagen = imagen_con_textura(64, 64)
        return {"status": "ok", "message": None, "prediction": "Tiña", "class_key": "ringworm",
                "confidence": 0.9,
                "top": [{"key": "ringworm", "label": "Tiña", "confidence": 0.9}],
                "analyzed_image": imagen, "gradcam_image": imagen}

    monkeypatch.setattr(model_utils_aislado, "predict_image", predict_falso)
    from backend.app.main import app
    with TestClient(app) as c:
        yield c


def test_rechaza_archivos_que_no_son_imagen(cliente):
    r = cliente.post("/predict", files={"file": ("nota.txt", b"hola", "text/plain")})
    assert r.status_code == 400


def test_respuesta_ok_respeta_el_contrato(cliente):
    r = cliente.post("/predict", files={"file": ("perro.jpg", jpeg(), "image/jpeg")})
    assert r.status_code == 200
    datos = r.json()
    assert set(datos) == CAMPOS
    assert datos["status"] == "ok"
    assert datos["analyzed_image"].startswith("data:image/jpeg;base64,")


def test_rechazo_respeta_el_contrato(model_utils_aislado, monkeypatch):
    monkeypatch.setattr(model_utils_aislado, "predict_image",
                        lambda s: model_utils_aislado.rejection("borrosa", "Imagen borrosa"))
    from backend.app.main import app
    with TestClient(app) as c:
        datos = c.post("/predict", files={"file": ("p.jpg", jpeg(), "image/jpeg")}).json()
    assert set(datos) == CAMPOS
    assert datos["status"] == "borrosa"
    assert datos["prediction"] is None and datos["confidence"] is None


def test_sirve_el_frontend_con_el_aviso_veterinario(cliente):
    r = cliente.get("/")
    assert r.status_code == 200
    assert "DermaDog" in r.text
    assert "veterinario" in r.text


@pytest.mark.hallazgo("H6")
def test_H6_rechaza_archivos_demasiado_grandes(cliente):
    """Un archivo de 25 MB debe rechazarse con 413 antes de llegar al modelo."""
    enorme = b"\xff\xd8\xff" + b"0" * (25 * 1024 * 1024)
    r = cliente.post("/predict", files={"file": ("grande.jpg", enorme, "image/jpeg")})
    assert r.status_code == 413


@pytest.mark.hallazgo("H8")
def test_H8_cors_no_combina_comodin_con_credenciales():
    from backend.app.main import app
    for capa in app.user_middleware:
        if capa.cls.__name__ == "CORSMiddleware":
            opciones = capa.kwargs
            assert not (opciones.get("allow_origins") == ["*"] and opciones.get("allow_credentials"))
