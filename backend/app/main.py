from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from . import model_utils
import io
import base64

app = FastAPI(title="Canine Dermal Disease Recognition")

# Allow CORS for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
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

# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    try:
        model_utils.initialize()
    except Exception as e:
        print(f"Failed to initialize model: {e}")

def to_data_url(image):
    if image is None:
        return None
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

# Plain `def` (not async): inference is CPU-bound, so FastAPI runs it in a
# worker thread instead of blocking the event loop for every other request
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = file.file.read()
        image_stream = io.BytesIO(contents)

        result = model_utils.predict_image(image_stream)

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
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Serve Frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
