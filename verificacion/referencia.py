"""Referencias fijas: la vara de medir del preprocesamiento y de las clases.

Definición única de evaluación (decisión del 2026-11-14, opción A): Resize(256) + CenterCrop(224),
la que ya usaba la app. Medida con verificacion/linea_base.py en valid y en test, rinde mejor que
Resize((224, 224)), la val_test_transforms del notebook. El modelo se entrenó con RandomResizedCrop
(70-100 % del área): el recorte central se parece más a esas imágenes que la foto completa deformada.

La app, la evaluación y la validación del entrenamiento deben producir exactamente este tensor.
"""
import torchvision.transforms as T

RESIZE = 256
IMG_SIZE = 224
MEDIA_IMAGENET = [0.485, 0.456, 0.406]
DESVIO_IMAGENET = [0.229, 0.224, 0.225]

# Orden de clases de ImageFolder en el entrenamiento
CLASS_TO_IDX = {
    "Dermatitis": 0,
    "Fungal_infections": 1,
    "Healthy": 2,
    "Hypersensitivity": 3,
    "demodicosis": 4,
    "ringworm": 5,
}


def transformacion_evaluacion():
    """Definición única de evaluación: Resize(256) + CenterCrop(224) + normalización de ImageNet."""
    return T.Compose([
        T.Resize(RESIZE),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(MEDIA_IMAGENET, DESVIO_IMAGENET),
    ])
