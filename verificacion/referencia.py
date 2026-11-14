"""Referencias fijas tomadas del notebook de entrenamiento (torchDogF.ipynb, celdas 7 y 25).

Son la vara de medir: la app debe darle al modelo exactamente lo que el modelo vio al entrenar.
"""
import torchvision.transforms as T

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
    """val_test_transforms del notebook: la que produjo el 0,94 de test."""
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(MEDIA_IMAGENET, DESVIO_IMAGENET),
    ])
