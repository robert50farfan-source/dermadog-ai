import torch
import torchvision.models as models
import json
import logging
import os
import threading
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from .gradcam import GradCAM
# Preprocesamiento único de evaluación (H1): no se redefine aquí
from .transforms import get_eval_transforms, model_crop_box

# Configuration - Easy to change later
MODEL_PATH = "backend/models/model_final_resnet50.pth"
CLASS_IDX_PATH = "backend/models/class_to_idx.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


class ModeloNoDisponible(RuntimeError):
    """El clasificador no se pudo cargar: la app no debe servir predicciones (H3)."""

# Spanish display names for the model classes (keys from class_to_idx.json)
CLASS_TRANSLATIONS = {
    "Dermatitis": "Dermatitis",
    "Fungal_infections": "Infecciones fúngicas",
    "Healthy": "Saludable",
    "Hypersensitivity": "Hipersensibilidad",
    "demodicosis": "Demodicosis",
    "ringworm": "Tiña"
}

# Blur check: sharpness of the sharpest regions, measured at a fixed size
BLUR_THRESHOLD = 25

# CLIP content check: labels with their Spanish names, and the ones that count as valid
CLIP_LABELS = {
    "skin disease on a dog": "Enfermedad de piel en perro",
    "dog skin close-up": "Primer plano de piel de perro",
    "human skin": "Piel humana",
    "cat fur": "Pelaje de gato",
    "object": "Objeto"
}
CLIP_DOG_LABELS = {"skin disease on a dog", "dog skin close-up"}
CLIP_MIN_DOG_PROB = 0.5

def sharpness_score(image, size=512, grid=4, percentile=90):
    # Variance of the Laplacian per tile, keeping a high percentile:
    # a sharp lesion on smooth skin or a dog in front of a blurred
    # background still scores high, while a fully blurred photo does not.
    # Resizing to a fixed size makes the score independent of resolution.
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    scale = size / max(w, h)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    gray = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=interpolation)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    h, w = laplacian.shape
    tiles = [
        laplacian[r * h // grid:(r + 1) * h // grid, c * w // grid:(c + 1) * w // grid].var()
        for r in range(grid) for c in range(grid)
    ]
    return float(np.percentile(tiles, percentile))

def center_crop_like_model(image, display_size=448):
    # Misma región que ve el modelo con get_eval_transforms() (constantes de transforms.py),
    # pero tomada de la resolución original para que la superposición de Grad-CAM se vea nítida.
    # El box admite coordenadas fraccionarias: PIL remuestrea esa región exacta, sin redondearla.
    box = model_crop_box(*image.size)
    return image.resize((display_size, display_size), resample=Image.BILINEAR, box=box)

def rejection(status, message):
    return {
        "status": status,
        "message": message,
        "prediction": None,
        "class_key": None,
        "confidence": None,
        "top": None,
        "analyzed_image": None,
        "gradcam_image": None,
    }

def load_class_mapping(path=CLASS_IDX_PATH):
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Using dummy classes.")
        # Dummy fallback
        return {"Disease A": 0, "Disease B": 1}
    
    with open(path, 'r') as f:
        class_to_idx = json.load(f)
    
    # Invert to idx_to_class
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return idx_to_class

def load_model(path=MODEL_PATH, num_classes=6):
    print(f"Loading model from {path}...")
    model = models.resnet50(pretrained=False) # Cargamos nuestros propios pesos

    # Ajusta la capa final para que coincida con la arquitectura guardada
    # Según las claves del state_dict: fc.1 (Linear), fc.3 (BN), fc.5 (Linear)
    # Es decir: 0:Dropout/Identity, 1:Linear, 2:ReLU, 3:BN, 4:Dropout, 5:Linear
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),             # Índice 0
        torch.nn.Linear(2048, 512),        # Índice 1
        torch.nn.ReLU(),                   # Índice 2
        torch.nn.BatchNorm1d(512),         # Índice 3
        torch.nn.Dropout(0.5),             # Índice 4
        torch.nn.Linear(512, num_classes)  # Índice 5
    )

    if os.path.exists(path):
        state_dict = torch.load(path, map_location=DEVICE)
        # Carga estricta (H3): si faltan o sobran claves, o no coinciden las formas,
        # load_state_dict lanza RuntimeError y se propaga. Nunca se cae a strict=False.
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            logger.exception("Los pesos de %s no coinciden con la arquitectura", path)
            raise
        logger.info("Modelo cargado en modo estricto desde %s", path)
    else:
        # Solo para tests y herramientas que necesitan la arquitectura sin pesos.
        # initialize() exige que el archivo exista antes de llamar aquí.
        logger.warning("No existe %s: modelo con pesos sin inicializar", path)

    model = model.to(DEVICE)
    model.eval()
    return model

# Variables globales que guardan el modelo y las clases
model = None
idx_to_class = None
grad_cam = None
clip_model = None
clip_processor = None
# Motivo del último fallo de carga del clasificador (None si no ha fallado)
load_error = None

# Serializa la carga del modelo (arranque o primera petición)
_init_lock = threading.Lock()
# Serializa la inferencia del ResNet y el Grad-CAM: comparten el mismo modelo,
# sus hooks y las activaciones/gradientes guardados (H4)
_inference_lock = threading.Lock()

def initialize():
    global model, idx_to_class, grad_cam, clip_model, clip_processor, load_error
    with _init_lock:
        if model is not None:
            # Otro hilo ya lo cargó mientras se esperaba el lock
            return
        try:
            # Sin pesos no se sirven predicciones: el modelo sin inicializar daría resultados basura
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"No existe el archivo de pesos {MODEL_PATH}")
            classes = load_class_mapping()
            loaded = load_model(num_classes=len(classes))
        except Exception as e:
            model = None
            grad_cam = None
            load_error = f"{type(e).__name__}: {e}"
            logger.error("No se pudo cargar el clasificador; /predict responderá 503. %s", load_error)
            raise ModeloNoDisponible(load_error) from e

        idx_to_class = classes
        # layer4 es el último bloque convolucional de ResNet
        grad_cam = GradCAM(loaded, loaded.layer4)
        model = loaded
        load_error = None

        # Inicializa CLIP
        logger.info("Cargando CLIP para validar el contenido...")
        try:
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            logger.info("CLIP cargado.")
        except Exception:
            logger.exception("No se pudo cargar CLIP; se omitirá la validación de contenido")

def ensure_model_loaded():
    """Garantiza que el clasificador está cargado o lanza ModeloNoDisponible (H3)."""
    if model is not None:
        return
    if load_error is not None:
        # La carga ya falló: no se reintenta en cada petición
        raise ModeloNoDisponible(load_error)
    initialize()
    if model is None:
        raise ModeloNoDisponible("El modelo no está cargado")

def predict_image(image_bytes):
    ensure_model_loaded()

    image = Image.open(image_bytes).convert('RGB')
    
    # Blur Detection
    score = sharpness_score(image)
    print(f"Image Sharpness Score: {score:.1f}") # Logging for debug

    if score < BLUR_THRESHOLD:
        return rejection("borrosa", "La imagen parece borrosa. Por favor, tome una fotografía más clara, con buena iluminación.")

    # CLIP Content Validation
    if clip_model and clip_processor:
        labels = list(CLIP_LABELS)
        inputs = clip_processor(text=labels, images=image, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)[0].tolist()

        # Add up every dog-skin label instead of only looking at the top one:
        # a close-up of hairless diseased skin often splits its probability
        # between the dog labels and "human skin"
        dog_prob = sum(p for label, p in zip(labels, probs) if label in CLIP_DOG_LABELS)
        print(f"CLIP dog-skin probability: {dog_prob:.2f} " + ", ".join(f"{l}={p:.2f}" for l, p in zip(labels, probs)))

        if dog_prob < CLIP_MIN_DOG_PROB:
            detected_class = max(
                ((label, p) for label, p in zip(labels, probs) if label not in CLIP_DOG_LABELS),
                key=lambda item: item[1],
            )[0]
            detected_class_es = CLIP_LABELS[detected_class]
            return rejection("no_valida", f"La imagen no parece ser adecuada. Se detectó: {detected_class_es}. Por favor suba una foto clara de la piel del perro.")

    transform = get_eval_transforms()
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Toma referencias locales: si otro hilo recarga el modelo, esta petición
    # usa de forma coherente el par (modelo, grad_cam) con el que empezó
    current_model, current_cam, classes = model, grad_cam, idx_to_class

    # Predicción y Grad-CAM bajo el mismo lock (H4): el modelo y sus hooks son estado compartido
    with _inference_lock:
        with torch.no_grad():
            outputs = current_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_class_idxs = torch.topk(probabilities, min(3, probabilities.shape[1]))

        class_idx = top_class_idxs[0, 0].item()
        confidence = top_probs[0, 0].item()

        # Umbral de confianza
        if confidence < 0.60:
            return rejection("baja_confianza", "No se puede determinar con suficiente confianza una enfermedad de la piel. La imagen no coincide claramente con los casos conocidos.")

        # Grad-CAM necesita gradientes, así que hace su propia pasada hacia delante y hacia atrás
        heatmap = current_cam.generate_heatmap(input_tensor, class_idx)

    predicted_label_en = classes.get(class_idx, "Unknown")
    predicted_label = CLASS_TRANSLATIONS.get(predicted_label_en, predicted_label_en)

    top = []
    for prob, idx in zip(top_probs[0].tolist(), top_class_idxs[0].tolist()):
        key = classes.get(idx, "Unknown")
        top.append({"key": key, "label": CLASS_TRANSLATIONS.get(key, key), "confidence": prob})

    # Superpone sobre la región exacta que analizó el modelo (recorte central);
    # si no, el mapa de calor queda desalineado en fotos no cuadradas
    analyzed_image = center_crop_like_model(image)
    result_image = GradCAM.overlay_heatmap(analyzed_image, heatmap)

    return {
        "status": "ok",
        "message": None,
        "prediction": predicted_label,
        "class_key": predicted_label_en,
        "confidence": confidence,
        "top": top,
        "analyzed_image": analyzed_image,
        "gradcam_image": result_image,
    }

