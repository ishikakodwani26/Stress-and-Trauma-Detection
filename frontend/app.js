import { WavRecorder } from './recorder.js';

// DOM Elements
const recordBtn = document.getElementById('recordBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const cancelBtn = document.getElementById('cancelBtn');
const timerDisplay = document.getElementById('timerDisplay');
const recordingPill = document.getElementById('recordingPill');
const canvas = document.getElementById('visualizerCanvas');
const audioPreviewCard = document.getElementById('audioPreviewCard');
const audioPlayer = document.getElementById('audioPlayer');
const audioMeta = document.getElementById('audioMeta');
const alertToast = document.getElementById('alertToast');
const alertMessage = document.getElementById('alertMessage');
const resultsContainer = document.getElementById('resultsContainer');
const emptyState = document.getElementById('emptyState');

// Canvas setup
const canvasCtx = canvas.getContext('2d');
let animationFrameId = null;
let currentRecordedBlob = null;
let currentWavUrl = null;

// Initialize WavRecorder instance
const recorder = new WavRecorder({
    onTimer: (elapsedMs) => {
        timerDisplay.textContent = formatDuration(elapsedMs);
    },
    onStateChange: ({ state, duration, sampleRate }) => {
        if (state === 'recording') {
            recordBtn.classList.add('recording');
            recordBtn.innerHTML = `
                <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24">
                    <rect x="6" y="6" width="12" height="12" rx="2"></rect>
                </svg>
                <span>Stop Recording</span>
            `;
            recordingPill.classList.add('active');
            cancelBtn.disabled = false;
            analyzeBtn.disabled = true;
            hideAlert();
        } else if (state === 'stopped') {
            recordBtn.classList.remove('recording');
            recordBtn.innerHTML = `
                <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24">
                    <circle cx="12" cy="12" r="8"></circle>
                </svg>
                <span>Re-Record</span>
            `;
            recordingPill.classList.remove('active');
            cancelBtn.disabled = false;
            analyzeBtn.disabled = false;
        } else if (state === 'idle') {
            resetRecorderUI();
        }
    }
});

// Start Visualizer Canvas Loop
function startVisualizer() {
    canvas.width = canvas.offsetWidth * window.devicePixelRatio;
    canvas.height = canvas.offsetHeight * window.devicePixelRatio;
    canvasCtx.scale(window.devicePixelRatio, window.devicePixelRatio);

    const width = canvas.offsetWidth;
    const height = canvas.offsetHeight;

    function draw() {
        animationFrameId = requestAnimationFrame(draw);

        canvasCtx.fillStyle = 'rgba(10, 15, 29, 0.4)';
        canvasCtx.fillRect(0, 0, width, height);

        const data = recorder.getVisualizerData();

        if (recorder.isRecording && data) {
            const { timeDomainData, frequencyData } = data;

            // Draw Frequency Bars (Backdrop)
            const barCount = 32;
            const barWidth = (width / barCount) - 3;
            for (let i = 0; i < barCount; i++) {
                const val = frequencyData[i * 2] / 255.0;
                const barHeight = val * (height * 0.7);
                const x = i * (barWidth + 3);
                const y = height - barHeight;

                const grad = canvasCtx.createLinearGradient(0, height, 0, y);
                grad.addColorStop(0, 'rgba(99, 102, 241, 0.15)');
                grad.addColorStop(1, 'rgba(6, 182, 212, 0.5)');

                canvasCtx.fillStyle = grad;
                canvasCtx.fillRect(x, y, barWidth, barHeight);
            }

            // Draw Oscilloscope Line (Foreground)
            canvasCtx.lineWidth = 2.5;
            canvasCtx.strokeStyle = '#06b6d4';
            canvasCtx.shadowBlur = 10;
            canvasCtx.shadowColor = 'rgba(6, 182, 212, 0.8)';
            canvasCtx.beginPath();

            const sliceWidth = width / timeDomainData.length;
            let x = 0;

            for (let i = 0; i < timeDomainData.length; i++) {
                const v = timeDomainData[i] / 128.0; // 0..2
                const y = (v * height) / 2;

                if (i === 0) {
                    canvasCtx.moveTo(x, y);
                } else {
                    canvasCtx.lineTo(x, y);
                }
                x += sliceWidth;
            }

            canvasCtx.stroke();
            canvasCtx.shadowBlur = 0; // reset
        } else {
            // Idle Flat Line with Soft Ambient Glow
            canvasCtx.lineWidth = 1.5;
            canvasCtx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
            canvasCtx.beginPath();
            canvasCtx.moveTo(0, height / 2);
            canvasCtx.lineTo(width, height / 2);
            canvasCtx.stroke();
        }
    }

    draw();
}

// Reset UI state
function resetRecorderUI() {
    recordBtn.classList.remove('recording');
    recordBtn.innerHTML = `
        <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="8"></circle>
        </svg>
        <span>Start Recording</span>
    `;
    recordingPill.classList.remove('active');
    timerDisplay.textContent = '00:00.0';
    cancelBtn.disabled = true;
    analyzeBtn.disabled = true;
    audioPreviewCard.classList.remove('visible');
    audioPlayer.src = '';
    currentRecordedBlob = null;
    if (currentWavUrl) {
        URL.revokeObjectURL(currentWavUrl);
        currentWavUrl = null;
    }
}

// Format duration in MM:SS.S
function formatDuration(ms) {
    const totalSeconds = ms / 1000;
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = Math.floor(totalSeconds % 60);
    const tenths = Math.floor((ms % 1000) / 100);
    return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${tenths}`;
}

// Show alert banner
function showAlert(message, type = 'error') {
    alertToast.className = `alert-toast visible ${type}`;
    alertMessage.textContent = message;
}

function hideAlert() {
    alertToast.className = 'alert-toast';
}

// Record button click handler
recordBtn.addEventListener('click', async () => {
    hideAlert();

    if (!recorder.isRecording) {
        try {
            await recorder.start();
        } catch (err) {
            console.error('Microphone access failed:', err);
            showAlert(`Microphone error: ${err.message || 'Permission denied'}`);
        }
    } else {
        try {
            const { blob, url, duration, sampleRate, totalSamples } = await recorder.stop();
            currentRecordedBlob = blob;
            currentWavUrl = url;

            // Update audio player preview
            audioPlayer.src = url;
            const sizeKb = (blob.size / 1024).toFixed(1);
            audioMeta.textContent = `RIFF WAV • ${sampleRate} Hz • 16-bit PCM • ${duration.toFixed(1)}s • ${sizeKb} KB`;
            audioPreviewCard.classList.add('visible');

        } catch (err) {
            console.error('Failed to stop recording:', err);
            showAlert(`Recording error: ${err.message}`);
        }
    }
});

// Cancel button click handler
cancelBtn.addEventListener('click', () => {
    recorder.cancel();
    resetRecorderUI();
    hideAlert();
});

// Analyze button click handler (sends .wav Blob to POST /predict/voice)
analyzeBtn.addEventListener('click', async () => {
    if (!currentRecordedBlob) {
        showAlert('No recorded audio available to analyze.');
        return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = `
        <div class="spinner"></div>
        <span>Analyzing Voice Streams...</span>
    `;
    hideAlert();

    try {
        const result = await WavRecorder.sendToVoiceEndpoint(currentRecordedBlob, '/predict/voice');
        renderPredictionResults(result);
        showAlert('Voice analysis completed successfully.', 'success');
    } catch (err) {
        console.error('Voice analysis failed:', err);
        showAlert(`Diagnostic failed: ${err.message}`);
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = `
            <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
            </svg>
            <span>Analyze Voice Stress & Trauma</span>
        `;
    }
});

// Render Prediction Results into Dashboard UI
function renderPredictionResults(data) {
    emptyState.style.display = 'none';

    const isElevated = data.predicted_label === 'elevated risk';
    const sviScore = data.metadata?.svi_score != null ? data.metadata.svi_score.toFixed(1) : 'N/A';
    const transcript = data.metadata?.transcript || '(No speech transcript detected)';
    const branches = data.metadata?.branch_scores || {};
    const probs = data.probabilities || {};

    const pElevated = probs.elevated_risk != null ? (probs.elevated_risk * 100).toFixed(1) : '--';
    const pLow = probs.low_risk != null ? (probs.low_risk * 100).toFixed(1) : '--';

    // Format Branch Rows
    const branchConfig = [
        {
            key: 'acoustic',
            name: 'Acoustic Stream',
            tag: 'openSMILE 88-D',
            colorClass: 'acoustic'
        },
        {
            key: 'emotion',
            name: 'Emotion Stream',
            tag: 'Wav2Vec2-XLS-R 1024-D',
            colorClass: 'emotion'
        },
        {
            key: 'nlp',
            name: 'ASR & NLP Stream',
            tag: 'IndicConformer + MuRIL 768-D',
            colorClass: 'nlp'
        }
    ];

    const branchHtml = branchConfig.map(cfg => {
        const branchData = branches[cfg.key] || {};
        const prob = branchData.probability != null ? (branchData.probability * 100).toFixed(1) : 0;
        const weight = branchData.weight != null ? (branchData.weight * 100).toFixed(0) : 0;
        const contrib = branchData.weighted_contribution != null ? (branchData.weighted_contribution * 100).toFixed(1) : 0;

        return `
            <div class="branch-row">
                <div class="branch-header">
                    <div class="branch-name">
                        <span>${cfg.name}</span>
                        <span class="branch-tag">${cfg.tag}</span>
                    </div>
                    <div class="branch-stats">
                        <span>Prob: <strong>${prob}%</strong></span> • 
                        <span>Weight: <strong>${weight}%</strong></span>
                    </div>
                </div>
                <div class="progress-track">
                    <div class="progress-bar ${cfg.colorClass}" style="width: ${prob}%;"></div>
                </div>
            </div>
        `;
    }).join('');

    resultsContainer.innerHTML = `
        <!-- SVI Hero Score Card -->
        <div class="svi-hero-card ${isElevated ? 'elevated-risk' : 'low-risk'}">
            <div class="svi-info">
                <h3>Stress Vulnerability Index (SVI)</h3>
                <div class="risk-badge ${isElevated ? 'elevated' : 'low'}">
                    ${isElevated ? '⚠️ Elevated Risk Detected' : '✓ Low Risk / Stable'}
                </div>
            </div>
            <div class="svi-gauge-wrap">
                <span class="svi-num" style="color: ${isElevated ? 'var(--danger)' : 'var(--success)'};">${sviScore}</span>
                <span class="svi-max">/100</span>
            </div>
        </div>

        <!-- Transcript Card -->
        <div class="transcript-card">
            <div class="transcript-title">IndicConformer Speech Transcript</div>
            <div class="transcript-text">"${escapeHtml(transcript)}"</div>
        </div>

        <!-- 3-Branch Breakdown -->
        <div class="branches-card">
            <div style="font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-secondary); margin-bottom: 4px;">
                Multimodal Branch Contributions
            </div>
            ${branchHtml}
        </div>

        <!-- Diagnostic Metadata -->
        <div class="metadata-card">
            <div class="meta-item">
                <span class="meta-key">P(Elevated Risk):</span>
                <span>${pElevated}%</span>
            </div>
            <div class="meta-item">
                <span class="meta-key">P(Low Risk):</span>
                <span>${pLow}%</span>
            </div>
            <div class="meta-item">
                <span class="meta-key">Fused Vector:</span>
                <span>${data.metadata?.feature_dimensions?.fused_total_dims || 1880} dimensions</span>
            </div>
            <div class="meta-item">
                <span class="meta-key">Modality:</span>
                <span>${data.modality || 'voice'}</span>
            </div>
        </div>
    `;
}

function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// Window resize listener for canvas sharpness
window.addEventListener('resize', () => {
    canvas.width = canvas.offsetWidth * window.devicePixelRatio;
    canvas.height = canvas.offsetHeight * window.devicePixelRatio;
    canvasCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
});

// Start visualizer on page load
startVisualizer();
