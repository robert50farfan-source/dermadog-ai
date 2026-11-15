"""Tests de H3 (carga estricta y 503), H4 (lock), H5 (hooks), H6 (413) y H8 (CORS).

No necesitan pesos reales, CLIP ni el dataset.
"""
import importlib
import io
import threading
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import model_utils
from backend.app.gradcam import GradCAM

RAIZ = Path(__file__).resolve().parents[1]


def imagen_jpeg(ancho=64, alto=64):
    rng = np.random.default_rng(0)
    imagen = Image.fromarray(rng.integers(0, 256, size=(alto, ancho, 3), dtype=np.uint8), "RGB")
    buffer = io.BytesIO()
    imagen.save(buffer, format="JPEG")
    return buffer.getvalue()


def resultado_ok():
    imagen = Image.new("RGB", (8, 8))
    return {"status": "ok", "message": None, "prediction": "Tiña", "class_key": "ringworm",
            "confidence": 0.9, "top": [{"key": "ringworm", "label": "Tiña", "confidence": 0.9}],
            "analyzed_image": imagen, "gradcam_image": imagen}


@pytest.fixture
def app_sin_modelo(monkeypatch):
    """La app con initialize() neutralizado (no carga ResNet ni CLIP)."""
    # main.py monta frontend/ con ruta relativa
    monkeypatch.chdir(RAIZ)
    monkeypatch.setattr(model_utils, "initialize", lambda: None)
    monkeypatch.setattr(model_utils, "model", None)
    monkeypatch.setattr(model_utils, "grad_cam", None)
    monkeypatch.setattr(model_utils, "load_error", None)
    from backend.app import main
    return main


# --- H3: carga estricta y 503 ---

def test_H3_pesos_con_forma_distinta_lanzan_error(tmp_path):
    modelo = model_utils.load_model(path="no-existe.pth", num_classes=6)
    ruta = tmp_path / "cuatro_clases.pth"
    torch.save(model_utils.load_model(path="no-existe.pth", num_classes=4).state_dict(), ruta)
    with pytest.raises(RuntimeError):
        model_utils.load_model(path=str(ruta), num_classes=6)
    assert modelo.fc[-1].out_features == 6


def test_H3_pesos_correctos_cargan(tmp_path):
    torch.manual_seed(0)
    original = model_utils.load_model(path="no-existe.pth", num_classes=6)
    ruta = tmp_path / "completo.pth"
    torch.save(original.state_dict(), ruta)
    cargado = model_utils.load_model(path=str(ruta), num_classes=6)
    for clave, valor in original.state_dict().items():
        assert torch.equal(valor, cargado.state_dict()[clave]), clave


def test_H3_initialize_sin_pesos_deja_la_app_sin_modelo(monkeypatch, tmp_path):
    monkeypatch.setattr(model_utils, "model", None)
    monkeypatch.setattr(model_utils, "grad_cam", None)
    monkeypatch.setattr(model_utils, "load_error", None)
    monkeypatch.setattr(model_utils, "MODEL_PATH", str(tmp_path / "no-existe.pth"))
    with pytest.raises(model_utils.ModeloNoDisponible):
        model_utils.initialize()
    assert model_utils.model is None
    assert "no-existe.pth" in model_utils.load_error
    # Una vez fallada la carga, predict_image no predice ni reintenta
    with pytest.raises(model_utils.ModeloNoDisponible):
        model_utils.predict_image(io.BytesIO(imagen_jpeg()))


def test_H3_predict_responde_503_si_el_modelo_no_cargo(app_sin_modelo, monkeypatch):
    monkeypatch.setattr(model_utils, "load_error", "RuntimeError: pesos incompatibles")
    with TestClient(app_sin_modelo.app) as cliente:
        r = cliente.post("/predict", files={"file": ("p.jpg", imagen_jpeg(), "image/jpeg")})
    assert r.status_code == 503
    assert "detail" in r.json()


# --- H6: límite de 10 MB ---

def test_H6_archivo_de_10_MB_mas_un_byte_da_413(app_sin_modelo, monkeypatch):
    llamado = []
    monkeypatch.setattr(model_utils, "predict_image", lambda s: llamado.append(1) or resultado_ok())
    datos = b"\xff\xd8\xff" + b"0" * (app_sin_modelo.MAX_UPLOAD_BYTES - 2)
    assert len(datos) == app_sin_modelo.MAX_UPLOAD_BYTES + 1
    with TestClient(app_sin_modelo.app) as cliente:
        r = cliente.post("/predict", files={"file": ("g.jpg", datos, "image/jpeg")})
    assert r.status_code == 413
    assert not llamado


def test_H6_archivo_de_10_MB_exactos_se_acepta(app_sin_modelo, monkeypatch):
    recibido = []

    def predict_falso(stream):
        recibido.append(len(stream.read()))
        return resultado_ok()

    monkeypatch.setattr(model_utils, "predict_image", predict_falso)
    datos = b"0" * app_sin_modelo.MAX_UPLOAD_BYTES
    with TestClient(app_sin_modelo.app) as cliente:
        r = cliente.post("/predict", files={"file": ("g.jpg", datos, "image/jpeg")})
    assert r.status_code == 200
    assert recibido == [app_sin_modelo.MAX_UPLOAD_BYTES]


def test_H6_read_limited_no_lee_mas_del_limite():
    from fastapi import HTTPException

    from backend.app.main import read_limited

    class Archivo:
        def __init__(self):
            self.leido = 0

        def read(self, n):
            # Flujo infinito: si no se cortara, el bucle no terminaría
            self.leido += n
            return b"0" * n

    class Subida:
        file = Archivo()

    with pytest.raises(HTTPException) as error:
        read_limited(Subida(), limit=1000)
    assert error.value.status_code == 413
    assert Subida.file.leido == 1001


# --- H8: CORS ---

def test_H8_cors_sin_comodin_ni_credenciales(app_sin_modelo):
    opciones = next(c.kwargs for c in app_sin_modelo.app.user_middleware
                    if c.cls.__name__ == "CORSMiddleware")
    assert "*" not in opciones["allow_origins"]
    assert opciones["allow_credentials"] is False


def test_H8_origenes_configurables_por_entorno(monkeypatch):
    monkeypatch.chdir(RAIZ)
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example, https://b.example")
    from backend.app import main
    try:
        main = importlib.reload(main)
        assert main.CORS_ORIGINS == ["https://a.example", "https://b.example"]
    finally:
        monkeypatch.delenv("CORS_ORIGINS")
        importlib.reload(main)


def test_H8_origen_desconocido_no_recibe_permiso(app_sin_modelo):
    with TestClient(app_sin_modelo.app) as cliente:
        r = cliente.options("/predict", headers={"Origin": "https://malicioso.example",
                                                 "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in r.headers


# --- H4 y H5: Grad-CAM ---

class RedPequena(torch.nn.Module):
    """Red mínima con una capa objetivo, para probar Grad-CAM rápido."""
    def __init__(self):
        super().__init__()
        self.capa = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU())
        self.salida = torch.nn.Linear(8, 4)

    def forward(self, x):
        return self.salida(self.capa(x).mean(dim=(2, 3)))


def test_H5_los_hooks_se_quitan_tras_cada_mapa():
    torch.manual_seed(0)
    red = RedPequena()
    cam = GradCAM(red, red.capa)
    cam.generate_heatmap(torch.randn(1, 3, 16, 16), 1)
    assert not red.capa._forward_hooks
    assert not red.capa._backward_hooks
    assert cam.activations is None and cam.gradients is None


def test_H5_los_hooks_se_quitan_aunque_falle():
    red = RedPequena()
    cam = GradCAM(red, red.capa)
    with pytest.raises(Exception):
        cam.generate_heatmap(torch.randn(1, 3, 16, 16), 99)  # clase fuera de rango
    assert not red.capa._forward_hooks
    assert not red.capa._backward_hooks


def test_H4_mapas_concurrentes_coinciden_con_los_secuenciales():
    torch.manual_seed(0)
    red = RedPequena().eval()
    cam = GradCAM(red, red.capa)
    entradas = [torch.randn(1, 3, 16, 16) for _ in range(8)]
    esperados = [cam.generate_heatmap(x, i % 4) for i, x in enumerate(entradas)]

    obtenidos = [None] * len(entradas)
    barrera = threading.Barrier(len(entradas))

    def trabajo(i):
        barrera.wait()
        for _ in range(5):
            obtenidos[i] = cam.generate_heatmap(entradas[i], i % 4)

    hilos = [threading.Thread(target=trabajo, args=(i,)) for i in range(len(entradas))]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    for esperado, obtenido in zip(esperados, obtenidos):
        assert np.allclose(esperado, obtenido)


def test_H4_predict_image_usa_el_lock_de_inferencia(monkeypatch):
    """Mientras un hilo tiene el lock, la predicción de otro espera."""
    monkeypatch.setattr(model_utils, "clip_model", None)
    monkeypatch.setattr(model_utils, "clip_processor", None)
    monkeypatch.setattr(model_utils, "idx_to_class", {i: str(i) for i in range(6)})

    class Modelo(torch.nn.Module):
        def forward(self, x):
            logits = torch.full((1, 6), -10.0)
            logits[0, 0] = 10.0
            return logits

    class CamNulo:
        def generate_heatmap(self, tensor, clase):
            return np.zeros((7, 7), dtype=np.float32)

    monkeypatch.setattr(model_utils, "model", Modelo())
    monkeypatch.setattr(model_utils, "grad_cam", CamNulo())
    terminado = threading.Event()
    datos = imagen_jpeg(320, 240)

    def trabajo():
        model_utils.predict_image(io.BytesIO(datos))
        terminado.set()

    with model_utils._inference_lock:
        hilo = threading.Thread(target=trabajo)
        hilo.start()
        assert not terminado.wait(timeout=1.0)
    hilo.join(timeout=30)
    assert terminado.is_set()
