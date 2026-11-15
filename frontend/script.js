// DOM Elements
const tabs = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');
const fileInput = document.getElementById('file-upload');
const dropZone = document.getElementById('drop-zone');
const video = document.getElementById('video-feed');
const canvas = document.getElementById('capture-canvas');
const captureBtn = document.getElementById('capture-btn');
const switchCameraBtn = document.getElementById('switch-camera-btn');
const torchBtn = document.getElementById('torch-btn');

// Result Elements
const inputPanel = document.querySelector('.input-panel');
const resultPanel = document.querySelector('.result-panel');
const loadingOverlay = document.getElementById('loading');
const resultsContainer = document.getElementById('results');
const predClassEl = document.getElementById('pred-class');
const predConfEl = document.getElementById('pred-conf');
const originalImgEl = document.getElementById('original-img');
const gradcamImgEl = document.getElementById('gradcam-img');
const diagnosisEl = document.getElementById('diagnosis');
const rejectionEl = document.getElementById('rejection');
const rejectionIconEl = document.getElementById('rejection-icon');
const rejectionTitleEl = document.getElementById('rejection-title');
const rejectionMessageEl = document.getElementById('rejection-message');
const resetBtn = document.getElementById('reset-btn');
const topListEl = document.getElementById('top-list');
const diseaseInfoEl = document.getElementById('disease-info');
const infoDescriptionEl = document.getElementById('info-description');
const infoSignsEl = document.getElementById('info-signs');
const infoContagiousEl = document.getElementById('info-contagious');
const infoContagiousLabelEl = document.getElementById('info-contagious-label');
const infoContagiousTextEl = document.getElementById('info-contagious-text');
const infoRecommendationEl = document.getElementById('info-recommendation');

const CONTAGIOUS_LABELS = {
    si: '¿Es contagiosa? Sí',
    no: '¿Es contagiosa? No',
    depende: '¿Es contagiosa? Depende de la causa'
};

// How each non-diagnosis status returned by the backend is presented
const REJECTIONS = {
    borrosa: { title: 'Imagen borrosa', icon: 'fa-eye-slash' },
    no_valida: { title: 'Imagen no válida', icon: 'fa-ban' },
    baja_confianza: { title: 'Resultado no concluyente', icon: 'fa-circle-question' }
};

let stream = null;
let currentFacingMode = 'environment'; // 'user' or 'environment'

// --- Tab Switching ---
tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));

        tab.classList.add('active');
        document.getElementById(`${tab.dataset.tab}-tab`).classList.add('active');

        if (tab.dataset.tab === 'camera') {
            startCamera();
        } else {
            stopCamera();
        }
    });
});

// --- File Upload Logic ---
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--primary)';
});

dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--border)';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--border)';
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    analyzeImage(file);
}

// --- Camera Logic ---
async function startCamera() {
    try {
        if (stream) stopCamera();

        // Ask for a high resolution: without it many phones default to
        // 640x480, too little detail for skin lesions
        const constraints = {
            video: {
                facingMode: currentFacingMode,
                width: { ideal: 1920 },
                height: { ideal: 1080 }
            }
        };

        stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
        setupTorch();
    } catch (err) {
        console.error("Error accessing camera:", err);
        let msg = "No se pudo acceder a la cámara.";

        if (err.name === 'NotAllowedError') {
            msg = "Permiso denegado. Por favor permite el acceso a la cámara en tu navegador.";
        } else if (err.name === 'NotFoundError') {
            msg = "No se encontró ninguna cámara en este dispositivo.";
        } else if (err.name === 'NotReadableError') {
            msg = "La cámara está siendo usada por otra aplicación.";
        } else if (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1' && window.location.protocol !== 'https:') {
            msg = "El acceso a la cámara requiere HTTPS o Localhost. Si estás accediendo desde otro dispositivo, necesitas un certificado SSL.";
        } else {
            msg = `Error: ${err.message}`;
        }

        alert(msg);
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        video.srcObject = null;
    }
    torchOn = false;
    torchBtn.classList.remove('active');
    torchBtn.classList.add('invisible');
}

// --- Torch (flashlight) ---
// Only offered when the browser and the active camera support it
// (mainly Chrome on Android, rear camera)
let torchOn = false;

function setupTorch() {
    const track = stream.getVideoTracks()[0];
    const capabilities = track.getCapabilities ? track.getCapabilities() : {};
    torchBtn.classList.toggle('invisible', !capabilities.torch);
}

torchBtn.addEventListener('click', async () => {
    if (!stream) return;
    const track = stream.getVideoTracks()[0];
    try {
        await track.applyConstraints({ advanced: [{ torch: !torchOn }] });
        torchOn = !torchOn;
        torchBtn.classList.toggle('active', torchOn);
    } catch (err) {
        console.error("Error toggling torch:", err);
    }
});

switchCameraBtn.addEventListener('click', () => {
    currentFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
    startCamera();
});

captureBtn.addEventListener('click', () => {
    if (!stream) return;

    // Capture only the centered square shown in the viewfinder,
    // so the photo sent is exactly what the user framed
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    const side = Math.min(vw, vh);
    canvas.width = side;
    canvas.height = side;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, (vw - side) / 2, (vh - side) / 2, side, side, 0, 0, side, side);

    // Convert to blob
    canvas.toBlob((blob) => {
        const file = new File([blob], "capture.jpg", { type: "image/jpeg" });

        stopCamera(); // Stop camera after capture
        analyzeImage(file);
    }, 'image/jpeg');
});

// --- API Interaction ---
// Mismo límite que el backend (MAX_UPLOAD_BYTES en backend/app/main.py)
const MAX_UPLOAD_MB = 10;

// Título de la tarjeta según el código de error del backend; el texto viene en `detail`
const ERROR_TITLES = {
    400: 'Archivo no válido',
    413: 'Imagen demasiado pesada',
    503: 'Servicio no disponible'
};

// Muestra un error en la misma tarjeta que los rechazos, con el botón "Nuevo Análisis"
function showError(title, message) {
    rejectionIconEl.className = 'fa-solid fa-triangle-exclamation rejection-icon';
    rejectionTitleEl.innerText = title;
    rejectionMessageEl.innerText = message;
    rejectionEl.classList.remove('hidden');
    diagnosisEl.classList.add('hidden');
    loadingOverlay.classList.add('hidden');
    resultsContainer.classList.remove('hidden');
}

async function analyzeImage(file) {
    // UI Transitions
    inputPanel.classList.add('hidden');
    resultPanel.classList.remove('hidden');
    loadingOverlay.classList.remove('hidden');
    resultsContainer.classList.add('hidden'); // Hide results until ready

    // Se avisa antes de subir: ante un archivo demasiado grande el servidor corta la
    // conexión, y en el celular el navegador suele ver un error de red, no el 413
    if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
        const mb = (file.size / (1024 * 1024)).toLocaleString('es', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
        showError(ERROR_TITLES[413], `La foto pesa ${mb} MB y el máximo es ${MAX_UPLOAD_MB} MB. ` +
            'Elija una foto más liviana o tómela con la pestaña Cámara.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    let response;
    try {
        response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
    } catch (error) {
        console.error(error);
        showError('Sin conexión', 'No se pudo conectar con el servidor. Revise su conexión e inténtelo de nuevo.');
        return;
    }

    if (!response.ok) {
        let detail = null;
        try {
            detail = (await response.json()).detail;
        } catch (_) {
            // Respuesta sin JSON (por ejemplo, de un proxy): se usa el mensaje genérico
        }
        showError(ERROR_TITLES[response.status] || 'No se pudo analizar',
            typeof detail === 'string' ? detail : 'Ocurrió un error inesperado. Inténtelo de nuevo.');
        return;
    }

    try {
        const data = await response.json();

        // Update UI
        if (data.status === 'ok') {
            predClassEl.innerText = data.prediction;
            predConfEl.innerText = `${(data.confidence * 100).toFixed(1)}%`;
            originalImgEl.src = data.analyzed_image;
            gradcamImgEl.src = data.gradcam_image;
            renderTopPredictions(data.top || []);
            renderDiseaseInfo(data.class_key);
            diagnosisEl.classList.remove('hidden');
            rejectionEl.classList.add('hidden');
        } else {
            const rejection = REJECTIONS[data.status] || { title: 'No se pudo analizar', icon: 'fa-triangle-exclamation' };
            rejectionIconEl.className = `fa-solid ${rejection.icon} rejection-icon`;
            rejectionTitleEl.innerText = rejection.title;
            rejectionMessageEl.innerText = data.message;
            rejectionEl.classList.remove('hidden');
            diagnosisEl.classList.add('hidden');
        }

        // Show Results
        loadingOverlay.classList.add('hidden');
        resultsContainer.classList.remove('hidden');

    } catch (error) {
        console.error(error);
        showError('No se pudo analizar', 'La respuesta del servidor no es válida. Inténtelo de nuevo.');
    }
}

// --- Result Rendering ---
function renderTopPredictions(top) {
    topListEl.replaceChildren();

    top.forEach((item, i) => {
        const pct = `${(item.confidence * 100).toFixed(1)}%`;

        const row = document.createElement('div');
        row.className = i === 0 ? 'top-row first' : 'top-row';

        const label = document.createElement('span');
        label.className = 'top-label';
        label.textContent = item.label;

        const bar = document.createElement('div');
        bar.className = 'top-bar';
        const fill = document.createElement('div');
        fill.className = 'top-bar-fill';
        fill.style.width = pct;
        bar.appendChild(fill);

        const value = document.createElement('span');
        value.className = 'top-value';
        value.textContent = pct;

        row.append(label, bar, value);
        topListEl.appendChild(row);
    });
}

function renderDiseaseInfo(classKey) {
    const info = DISEASE_INFO[classKey];
    if (!info) {
        diseaseInfoEl.classList.add('hidden');
        return;
    }

    infoDescriptionEl.textContent = info.description;

    infoSignsEl.replaceChildren(...info.signs.map(sign => {
        const li = document.createElement('li');
        li.textContent = sign;
        return li;
    }));

    infoContagiousEl.className = `contagious ${info.contagious}`;
    infoContagiousLabelEl.textContent = CONTAGIOUS_LABELS[info.contagious];
    infoContagiousTextEl.textContent = info.contagiousText;
    infoRecommendationEl.textContent = info.recommendation;

    diseaseInfoEl.classList.remove('hidden');
}

// --- Reset Logic ---
resetBtn.addEventListener('click', resetState);

function resetState() {
    resultPanel.classList.add('hidden');
    inputPanel.classList.remove('hidden');
    fileInput.value = '';

    // Reset tabs to default (Upload) or keep current? 
    // Let's reset to Upload mainly because camera was stopped.
    // If user was on camera tab, they might want to take another photo.
    // Let's check which tab is active.
    const activeTab = document.querySelector('.tab-btn.active').dataset.tab;
    if (activeTab === 'camera') {
        startCamera();
    }
}
