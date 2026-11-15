from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from . import model_utils
import io
import base64
import logging
import os

logger = logging.getLogger(__name__)

app = FastAPI(title="Canine Dermal Disease Recognition")

# Tamaño máximo del archivo subido a /predict (H6)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
# Bloque de lectura: nunca se tiene en memoria más de MAX_UPLOAD_BYTES + un bloque
READ_CHUNK_BYTES = 1024 * 1024
# Holgura para las cabeceras multipart al comparar con Content-Length
MULTIPART_OVERHEAD_BYTES = 64 * 1024

# CORS (H8): el frontend se sirve desde este mismo servidor y llama a /predict con
# una ruta relativa, así que el navegador no necesita CORS. Solo se admiten los
# orígenes listados (por defecto, desarrollo local) y sin credenciales: la API no usa cookies.
# Para otros orígenes: CORS_ORIGINS="https://ejemplo.com,https://otro.com"
DEFAULT_CORS_ORIGINS = "http://localhost:8000,http://127.0.0.1:8000"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Make the browser revalidate frontend files on every load, so changes to
# script.js / styles.css show up without a stale cached copy
@app.middleware("http")
async def no_cache_static(request, call_next):
    response = await call_next(request)
    if request.url.path != "/predict":
        response.headers["Cache-Control"] = "no-cache"
    return response

# Rechazo temprano (H6): si Content-Length ya anuncia un cuerpo demasiado grande,
# se responde 413 antes de recibirlo y de que se procese el multipart.
# Si no hay Content-Length, el control exacto lo hace /predict al leer el archivo.
@app.middleware("http")
async def limit_upload_size(request, call_next):
    if request.url.path == "/predict" and request.method == "POST":
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES:
            return JSONResponse(status_code=413, content={"detail": too_large_message()})
    return await call_next(request)

def too_large_message():
    return f"El archivo supera el tamaño máximo de {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."

# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    try:
        model_utils.initialize()
    except Exception:
        # La app arranca (sirve el frontend), pero /predict responderá 503 (H3)
        logger.exception("No se pudo inicializar el modelo")

def to_data_url(image):
    if image is None:
        return None
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

def read_limited(upload, limit=MAX_UPLOAD_BYTES):
    """Lee el archivo en bloques; lanza 413 en cuanto se pasa de `limit` bytes (H6)."""
    buffer = io.BytesIO()
    size = 0
    while True:
        chunk = upload.file.read(min(READ_CHUNK_BYTES, limit + 1 - size))
        if not chunk:
            break
        size += len(chunk)
        if size > limit:
            raise HTTPException(status_code=413, detail=too_large_message())
        buffer.write(chunk)
    buffer.seek(0)
    return buffer

# Plain `def` (not async): inference is CPU-bound, so FastAPI runs it in a
# worker thread instead of blocking the event loop for every other request
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_stream = read_limited(file)

    try:
        result = model_utils.predict_image(image_stream)
    except model_utils.ModeloNoDisponible as e:
        # Nunca se predice con pesos incorrectos o sin cargar (H3)
        logger.error("Predicción rechazada: modelo no disponible (%s)", e)
        raise HTTPException(status_code=503, detail="El modelo no está disponible. Inténtelo más tarde.")
    except Exception as e:
        logger.exception("Error al procesar /predict")
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content={
        "status": result["status"],
        "message": result["message"],
        "prediction": result["prediction"],
        "class_key": result["class_key"],
        "confidence": None if result["confidence"] is None else float(result["confidence"]),
        "top": result["top"],
        "analyzed_image": to_data_url(result["analyzed_image"]),
        "gradcam_image": to_data_url(result["gradcam_image"]),
    })

# Serve Frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
