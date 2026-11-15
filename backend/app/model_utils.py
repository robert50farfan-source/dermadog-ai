import torch
import torchvision.models as models
import json
import os
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
        # Contempla el envoltorio de DataParallel si las claves empiezan por 'module.'
        # (no se observó al inspeccionar los pesos, pero es buena práctica)
        # state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        try:
            model.load_state_dict(state_dict)
            print("Model loaded successfully.")
        except RuntimeError as e:
            print(f"Error loading state_dict: {e}")
            print("Attempting strict=False load (Use with caution)...")
            model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Warning: Model file {path} not found. Using uninitialized weights (garbage predictions).")
    
    model = model.to(DEVICE)
    model.eval()
    return model

# Variables globales que guardan el modelo y las clases
model = None
idx_to_class = None
grad_cam = None
clip_model = None
clip_processor = None

def initialize():
    global model, idx_to_class, grad_cam, clip_model, clip_processor
    idx_to_class = load_class_mapping()
    num_classes = len(idx_to_class)
    model = load_model(num_classes=num_classes)
    
    # Inicializa CLIP
    print("Loading CLIP model for content validation...")
    try:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("CLIP model loaded successfully.")
    except Exception as e:
        print(f"Failed to load CLIP: {e}")
    
    # Inicializa Grad-CAM
    # layer4 es el último bloque convolucional de ResNet
    grad_cam = GradCAM(model, model.layer4)

def predict_image(image_bytes):
    global model, idx_to_class, grad_cam
    
    if model is None:
        initialize()
        
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
    
    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_class_idxs = torch.topk(probabilities, min(3, probabilities.shape[1]))

    class_idx = top_class_idxs[0, 0].item()
    confidence = top_probs[0, 0].item()
    predicted_label_en = idx_to_class.get(class_idx, "Unknown")

    # Threshold Check
    if confidence < 0.60:
        return rejection("baja_confianza", "No se puede determinar con suficiente confianza una enfermedad de la piel. La imagen no coincide claramente con los casos conocidos.")

    predicted_label = CLASS_TRANSLATIONS.get(predicted_label_en, predicted_label_en)

    top = []
    for prob, idx in zip(top_probs[0].tolist(), top_class_idxs[0].tolist()):
        key = idx_to_class.get(idx, "Unknown")
        top.append({"key": key, "label": CLASS_TRANSLATIONS.get(key, key), "confidence": prob})

    # Grad-CAM necesita gradientes, así que hace su propia pasada hacia delante y hacia atrás
    heatmap = grad_cam.generate_heatmap(input_tensor, class_idx)

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

