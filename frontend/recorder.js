/**
 * WavRecorder - Web Audio API Audio Recorder & Pure JavaScript WAV Encoder.
 * 
 * Captures raw Float32 PCM samples from microphone via Web Audio API
 * and manually encodes them into a standard 44-byte RIFF 16-bit PCM WAV Blob.
 * 
 * Bypasses MediaRecorder entirely so audio is 100% compliant WAV with ZERO
 * dependency on backend FFmpeg or container demuxers.
 */

/**
 * Manually encode Float32 PCM samples into a standard 16-bit PCM WAV Blob.
 * 
 * @param {Float32Array} samples - Audio samples normalized between -1.0 and 1.0.
 * @param {number} sampleRate - Audio sample rate (e.g. 44100 or 48000).
 * @param {number} [numChannels=1] - Number of audio channels (1 for mono).
 * @returns {Blob} Standard audio/wav Blob with 44-byte RIFF header.
 */
export function encodeWAV(samples, sampleRate, numChannels = 1) {
    const bytesPerSample = 2; // 16-bit signed integer
    const blockAlign = numChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = samples.length * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    // Helper to write ASCII strings to DataView
    function writeString(offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    // 1. RIFF Chunk Descriptor
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true); // ChunkSize = 36 + SubChunk2Size
    writeString(8, 'WAVE');

    // 2. fmt Sub-chunk
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);             // Subchunk1Size = 16 for PCM
    view.setUint16(20, 1, true);              // AudioFormat = 1 (linear PCM)
    view.setUint16(22, numChannels, true);    // NumChannels (1 = mono, 2 = stereo)
    view.setUint32(24, sampleRate, true);     // SampleRate (e.g. 44100, 48000)
    view.setUint32(28, byteRate, true);       // ByteRate = SampleRate * NumChannels * BitsPerSample/8
    view.setUint16(32, blockAlign, true);     // BlockAlign = NumChannels * BitsPerSample/8
    view.setUint16(34, 16, true);             // BitsPerSample = 16

    // 3. data Sub-chunk
    writeString(36, 'data');
    view.setUint32(40, dataSize, true);       // Subchunk2Size = NumSamples * NumChannels * 2

    // 4. Write 16-bit PCM samples with [-1.0, 1.0] clamping
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        const int16 = s < 0 ? s * 0x8000 : s * 0x7FFF;
        view.setInt16(offset, int16, true);
    }

    return new Blob([view], { type: 'audio/wav' });
}

/**
 * WavRecorder class wrapping AudioContext microphone input and PCM recording.
 */
export class WavRecorder {
    constructor(options = {}) {
        this.options = {
            bufferSize: 4096,
            preferSampleRate: null,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
            ...options
        };

        this.audioContext = null;
        this.mediaStream = null;
        this.sourceNode = null;
        this.processorNode = null;
        this.analyserNode = null;

        this.isRecording = false;
        this.recordedChunks = [];
        this.totalSamples = 0;
        this.sampleRate = 44100;
        this.startTime = null;
        this.timerInterval = null;

        // Callbacks
        this.onStateChange = options.onStateChange || null;
        this.onTimer = options.onTimer || null;
        this.onAudioProcess = options.onAudioProcess || null;
    }

    /**
     * Start capturing microphone audio.
     */
    async start() {
        if (this.isRecording) {
            console.warn('WavRecorder is already recording.');
            return;
        }

        // Request microphone access
        const constraints = {
            audio: {
                channelCount: 1,
                echoCancellation: this.options.echoCancellation,
                noiseSuppression: this.options.noiseSuppression,
                autoGainControl: this.options.autoGainControl
            }
        };

        if (this.options.preferSampleRate) {
            constraints.audio.sampleRate = this.options.preferSampleRate;
        }

        this.mediaStream = await navigator.mediaDevices.getUserMedia(constraints);

        // Initialize AudioContext
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        this.audioContext = new AudioContextClass();
        this.sampleRate = this.audioContext.sampleRate;

        // Create MediaStream source
        this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);

        // Create AnalyserNode for real-time visualization
        this.analyserNode = this.audioContext.createAnalyser();
        this.analyserNode.fftSize = 256;
        this.analyserNode.smoothingTimeConstant = 0.8;

        // Create ScriptProcessorNode for raw PCM extraction
        this.processorNode = this.audioContext.createScriptProcessor(
            this.options.bufferSize,
            1, // mono input
            1  // mono output
        );

        this.recordedChunks = [];
        this.totalSamples = 0;
        this.isRecording = true;
        this.startTime = Date.now();

        this.processorNode.onaudioprocess = (e) => {
            if (!this.isRecording) return;
            const inputData = e.inputBuffer.getChannelData(0);
            const chunk = new Float32Array(inputData.length);
            chunk.set(inputData);
            this.recordedChunks.push(chunk);
            this.totalSamples += inputData.length;

            if (this.onAudioProcess) {
                this.onAudioProcess(inputData);
            }
        };

        // Wire audio graph: Source -> Analyser -> Processor -> Destination
        // Note: Connecting to destination is required in WebKit/Blink for onaudioprocess to run
        this.sourceNode.connect(this.analyserNode);
        this.analyserNode.connect(this.processorNode);
        this.processorNode.connect(this.audioContext.destination);

        // Start timer
        this._startTimer();

        if (this.onStateChange) {
            this.onStateChange({
                state: 'recording',
                sampleRate: this.sampleRate
            });
        }
    }

    /**
     * Stop capturing and return encoded 16-bit PCM WAV Blob.
     * 
     * @returns {Promise<{ blob: Blob, url: string, duration: number, sampleRate: number, totalSamples: number }>}
     */
    async stop() {
        if (!this.isRecording) {
            throw new Error('WavRecorder is not currently recording.');
        }

        this.isRecording = false;
        this._stopTimer();

        // Stop microphone tracks so recording indicator on browser turns off
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach((track) => track.stop());
        }

        // Disconnect audio nodes
        if (this.processorNode) {
            this.processorNode.disconnect();
            this.processorNode.onaudioprocess = null;
        }
        if (this.analyserNode) {
            this.analyserNode.disconnect();
        }
        if (this.sourceNode) {
            this.sourceNode.disconnect();
        }
        if (this.audioContext && this.audioContext.state !== 'closed') {
            await this.audioContext.close();
        }

        // Merge recorded Float32 chunks into single continuous array
        const mergedSamples = new Float32Array(this.totalSamples);
        let offset = 0;
        for (const chunk of this.recordedChunks) {
            mergedSamples.set(chunk, offset);
            offset += chunk.length;
        }

        // Encode to standard 16-bit PCM WAV
        const wavBlob = encodeWAV(mergedSamples, this.sampleRate, 1);
        const wavUrl = URL.createObjectURL(wavBlob);
        const durationSec = this.totalSamples / this.sampleRate;

        if (this.onStateChange) {
            this.onStateChange({
                state: 'stopped',
                duration: durationSec,
                sampleRate: this.sampleRate,
                totalSamples: this.totalSamples
            });
        }

        return {
            blob: wavBlob,
            url: wavUrl,
            duration: durationSec,
            sampleRate: this.sampleRate,
            totalSamples: this.totalSamples
        };
    }

    /**
     * Cancel current recording without producing output.
     */
    cancel() {
        this.isRecording = false;
        this._stopTimer();

        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach((track) => track.stop());
        }
        if (this.processorNode) {
            this.processorNode.disconnect();
            this.processorNode.onaudioprocess = null;
        }
        if (this.analyserNode) this.analyserNode.disconnect();
        if (this.sourceNode) this.sourceNode.disconnect();
        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close();
        }

        this.recordedChunks = [];
        this.totalSamples = 0;

        if (this.onStateChange) {
            this.onStateChange({ state: 'idle' });
        }
    }

    /**
     * Get real-time audio visualizer frequency and time-domain data.
     * 
     * @returns {{ frequencyData: Uint8Array, timeDomainData: Uint8Array } | null}
     */
    getVisualizerData() {
        if (!this.analyserNode) return null;
        const frequencyData = new Uint8Array(this.analyserNode.frequencyBinCount);
        const timeDomainData = new Uint8Array(this.analyserNode.frequencyBinCount);
        this.analyserNode.getByteFrequencyData(frequencyData);
        this.analyserNode.getByteTimeDomainData(timeDomainData);
        return { frequencyData, timeDomainData };
    }

    /**
     * Post the recorded WAV Blob directly to the backend /predict/voice endpoint.
     * 
     * @param {Blob} wavBlob - The standard WAV Blob produced by stop() or encodeWAV().
     * @param {string} [endpointUrl='/predict/voice'] - Backend API URL.
     * @returns {Promise<Object>} The PredictionResponse JSON from the backend.
     */
    static async sendToVoiceEndpoint(wavBlob, endpointUrl = '/predict/voice') {
        if (!wavBlob || !(wavBlob instanceof Blob)) {
            throw new Error('A valid WAV Blob is required.');
        }

        const formData = new FormData();
        formData.append('audio', wavBlob, 'recording.wav');

        const response = await fetch(endpointUrl, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            let errorDetail = `HTTP ${response.status} ${response.statusText}`;
            try {
                const errorJson = await response.json();
                errorDetail = errorJson.detail || errorJson.message || errorDetail;
            } catch (_) {
                // Ignore JSON parse error on non-JSON response
            }
            throw new Error(errorDetail);
        }

        return await response.json();
    }

    _startTimer() {
        this._stopTimer();
        this.timerInterval = setInterval(() => {
            if (!this.isRecording || !this.startTime) return;
            const elapsedMs = Date.now() - this.startTime;
            if (this.onTimer) {
                this.onTimer(elapsedMs);
            }
        }, 50);
    }

    _stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
    }
}

// Attach to window for global non-module usage if desired
if (typeof window !== 'undefined') {
    window.WavRecorder = WavRecorder;
    window.encodeWAV = encodeWAV;
}
