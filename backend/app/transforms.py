"""Definición única del preprocesamiento de evaluación de DermaDog (hallazgo H1).

La usan la app (model_utils.py), la evaluación y la validación del entrenamiento.
Debe producir exactamente el mismo tensor que verificacion/referencia.py:
Resize(256) + CenterCrop(224) + ToTensor + Normalize(ImageNet).
Decisión del 2026-11-14: no se cambia sin una nueva medición aprobada por el humano.
"""
import torchvision.transforms as T

# Tamaños: el lado corto se lleva a RESIZE_SIZE y luego se recorta un cuadrado central de CROP_SIZE
RESIZE_SIZE = 256
CROP_SIZE = 224

# Normalización con la media y la desviación estándar de ImageNet (pesos preentrenados de ResNet50)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_eval_transforms():
    """Preprocesamiento de evaluación: Resize(256) + CenterCrop(224) + ToTensor + Normalize(ImageNet)."""
    return T.Compose([
        T.Resize(RESIZE_SIZE),
        T.CenterCrop(CROP_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def model_crop_box(width, height):
    """Región (left, top, right, bottom), en coordenadas de la imagen original, que ve el modelo.

    Replica el redondeo de torchvision:
    - Resize(int): el lado corto pasa a RESIZE_SIZE y el largo se trunca con int().
    - CenterCrop: el desplazamiento es int(round((tamaño - CROP_SIZE) / 2)).
    El recorte en la imagen redimensionada se lleva de vuelta a la original con la escala de cada eje,
    que es la misma correspondencia de píxeles que usa PIL al redimensionar.
    """
    if width <= height:
        new_w, new_h = RESIZE_SIZE, int(RESIZE_SIZE * height / width)
    else:
        new_w, new_h = int(RESIZE_SIZE * width / height), RESIZE_SIZE
    crop_left = int(round((new_w - CROP_SIZE) / 2.0))
    crop_top = int(round((new_h - CROP_SIZE) / 2.0))
    scale_x = width / new_w
    scale_y = height / new_h
    return (
        crop_left * scale_x,
        crop_top * scale_y,
        (crop_left + CROP_SIZE) * scale_x,
        (crop_top + CROP_SIZE) * scale_y,
    )
