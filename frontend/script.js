// DOM Elements
const tabs = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');
const fileInput = document.getElementById('file-upload');
const dropZone = document.getElementById('drop-zone');
const video = document.getElementById('video-feed');
const canvas = document.getElementById('capture-canvas');
const captureBtn = document.getElementById('capture-btn');
const switchCameraBtn = document.getElementById('switch-camera-btn');

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

        const constraints = {
            video: {
                facingMode: currentFacingMode
            }
        };

        stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
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
}

switchCameraBtn.addEventListener('click', () => {
    currentFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
    startCamera();
});

captureBtn.addEventListener('click', () => {
    if (!stream) return;

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw current frame
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to blob
    canvas.toBlob((blob) => {
        const file = new File([blob], "capture.jpg", { type: "image/jpeg" });

        stopCamera(); // Stop camera after capture
        analyzeImage(file);
    }, 'image/jpeg');
});

// --- API Interaction ---
async function analyzeImage(file) {
    // UI Transitions
    inputPanel.classList.add('hidden');
    resultPanel.classList.remove('hidden');
    loadingOverlay.classList.remove('hidden');
    resultsContainer.classList.add('hidden'); // Hide results until ready

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("Error en la predicción");

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
        alert("Ocurrió un error. Inténtalo de nuevo.");
        resetState();
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
