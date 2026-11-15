# DermaDog AI

Aplicación web que reconoce enfermedades de la piel en perros a partir de una fotografía, usando deep learning.

> ⚠️ Es una herramienta de apoyo con fines académicos. **No reemplaza el diagnóstico de un médico veterinario.**

Proyecto final del Diplomado en Inteligencia Artificial y Ciencia de Datos — FIRNT.

## Qué hace

El usuario sube una foto o la toma con la cámara, y la aplicación:

1. **Revisa que la foto sea nítida.** Mide la nitidez (varianza del Laplaciano) en las zonas más definidas de la imagen.
2. **Revisa que sea piel de perro.** Usa [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) para rechazar piel humana, gatos u objetos.
3. **Clasifica la imagen** con una **ResNet50** ajustada (fine-tuning) en 6 clases:
   - Dermatitis
   - Infecciones fúngicas
   - Hipersensibilidad
   - Demodicosis
   - Tiña
   - Saludable
4. **Muestra el resultado:**
   - la enfermedad más probable y su confianza;
   - las 3 clases más probables;
   - un mapa de calor **Grad-CAM** con las zonas en las que se fijó el modelo;
   - una ficha informativa: signos, si es contagiosa y una recomendación.

Si la confianza es menor al 60 %, el resultado se informa como **no concluyente** en lugar de dar un diagnóstico.

## Estructura

```
appDog/
├── backend/
│   ├── app/
│   │   ├── main.py          # API FastAPI (/predict) y servidor del frontend
│   │   ├── model_utils.py   # Carga de modelos, validaciones y predicción
│   │   └── gradcam.py       # Mapa de calor Grad-CAM
│   ├── models/
│   │   ├── model_final_resnet50.pth   # Pesos del modelo (Git LFS)
│   │   └── class_to_idx.json          # Clases del modelo
│   └── requirements.txt
└── frontend/
    ├── index.html
    ├── script.js
    ├── diseases.js          # Ficha informativa de cada enfermedad
    └── styles.css
```

## Requisitos

- Python 3.11
- [Git LFS](https://git-lfs.com/), para descargar el archivo del modelo
- Internet la primera vez que se ejecuta, para descargar CLIP (unos 600 MB)

## Instalación y ejecución

```bash
git lfs install
git clone <url-del-repositorio>
cd <carpeta-del-repositorio>

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r backend/requirements.txt
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Luego abre **http://localhost:8000** en el navegador.

Los comandos deben ejecutarse desde la carpeta raíz del proyecto, porque las rutas al modelo y al frontend son relativas.

> La cámara del navegador solo funciona en `localhost` o con HTTPS. Si entras desde otro dispositivo de la red, usa la opción de subir archivo.

## API

`POST /predict` recibe una imagen (`multipart/form-data`, campo `file`) y devuelve:

```json
{
  "status": "ok",
  "prediction": "Demodicosis",
  "class_key": "demodicosis",
  "confidence": 0.917,
  "top": [{ "key": "demodicosis", "label": "Demodicosis", "confidence": 0.917 }],
  "message": null,
  "analyzed_image": "data:image/jpeg;base64,...",
  "gradcam_image": "data:image/jpeg;base64,..."
}
```

`status` puede ser:

| `status` | Significado |
|---|---|
| `ok` | Hay diagnóstico. |
| `borrosa` | La foto está borrosa. |
| `no_valida` | La foto no parece piel de perro. |
| `baja_confianza` | El modelo no llegó al 60 % de confianza. |

En los tres casos distintos de `ok`, el motivo viene en `message`.

Errores HTTP (cuerpo `{"detail": "..."}`, sin los campos anteriores):

| Código | Cuándo |
|---|---|
| `400` | El archivo no es una imagen (`Content-Type` distinto de `image/*`). |
| `413` | El archivo supera 10 MB. |
| `503` | El modelo no se pudo cargar (pesos ausentes o que no coinciden con la arquitectura). La app no predice con pesos incorrectos; el motivo queda en el log. |
| `500` | Error inesperado al procesar la imagen. |

CORS: el frontend se sirve desde el mismo servidor, así que no lo necesita. Por defecto solo se admiten
`http://localhost:8000` y `http://127.0.0.1:8000`, sin credenciales. Para otros orígenes, defina
`CORS_ORIGINS` con una lista separada por comas (p. ej. `CORS_ORIGINS=https://midominio.com`).

## Limitaciones conocidas

- El modelo confunde algunos casos de **tiña** con piel **saludable**.
- La **sarna sarcóptica** no es una de las clases del modelo, así que se clasifica como la clase más parecida.

## Autor

Luis Robert Farfan Sivila
