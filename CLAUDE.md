# DermaDog AI — contexto para Claude Code

App web que clasifica enfermedades de la piel en perros a partir de una foto.
Backend FastAPI + ResNet50 (PyTorch), validación previa con nitidez (Laplaciano) y CLIP,
explicación con Grad-CAM. Frontend HTML/CSS/JS sin frameworks. Despliegue con Docker en Dokploy.

## Clases (orden de class_to_idx.json, no cambiar)
0 Dermatitis · 1 Fungal_infections · 2 Healthy · 3 Hypersensitivity · 4 demodicosis · 5 ringworm

## Estructura
- backend/app/main.py — API (/predict) y servidor del frontend
- backend/app/model_utils.py — carga de modelos, validaciones, predicción
- backend/app/gradcam.py — mapa de calor
- backend/models/ — pesos (Git LFS) y class_to_idx.json. NO se modifican: solo el humano reemplaza el modelo.
- frontend/ — interfaz
- training/ — (a crear) código de entrenamiento reproducible, basado en notebooks/torchDogF.ipynb
- data/ — dataset de Kaggle (no se sube a Git). data/test solo se usa en verificacion/linea_base.py
- verificacion/ — controles automáticos y línea base. NO se modifican: son la vara de medir.

## Reglas
1. Un solo preprocesamiento de evaluación, compartido por entrenamiento, evaluación y app.
   La referencia es la del entrenamiento: Resize((224, 224)) + ToTensor + Normalize(ImageNet).
2. Ninguna métrica se escribe a mano: salen de scikit-learn o de los scripts de verificacion/.
3. El mejor modelo se guarda con copy.deepcopy(model.state_dict()) junto con época y métrica.
4. Todo cambio debe dejar verde: `python -m pytest verificacion/tests -m "not hallazgo" -q`.
5. Nada se despliega ni se hace push a main sin aprobación humana.
6. Comentarios y mensajes al usuario en español.
7. Si no puedes verificar algo ejecutándolo, dilo; no lo afirmes.

## Hallazgos abiertos (auditoría 2026-11-14)
- H1 (alta) La app preprocesa con Resize(256)+CenterCrop(224); el modelo se entrenó y evaluó con Resize((224,224)).
- H2 (alta) Notebook: state_dict().copy() es copia superficial; model_final_resnet50.pth es la última época, no la mejor.
- H3 (media) load_model() cae a strict=False en silencio si los pesos no coinciden.
- H4 (media) Grad-CAM comparte estado (hooks, activaciones) entre peticiones concurrentes sin lock.
- H5 (baja) gradcam.py usa register_backward_hook (obsoleto); usar register_full_backward_hook.
- H6 (baja) /predict no limita el tamaño del archivo.
- H7 (media) Umbrales (nitidez 25, CLIP 0,5, confianza 0,60) sin datos de calibración documentados.
- H8 (baja) CORS allow_origins=["*"] con allow_credentials=True.

Los tests de verificacion/tests marcados `hallazgo` fallan hasta que se corrija el hallazgo correspondiente.

## Comandos
- Tests rápidos: `python -m pytest verificacion/tests -m "not hallazgo" -q`
- Un hallazgo: `python -m pytest verificacion/tests -m hallazgo -k H1 -q`
- Línea base: `python verificacion/linea_base.py --datos data --salida verificacion/resultados/actual.json`
- Comparar: `python verificacion/comparar.py verificacion/linea_base.json verificacion/resultados/actual.json`
- App local: `uvicorn backend.app.main:app --port 8000`
