export class InferenceUploader {
    constructor() {
        this.uploadForm = document.getElementById('uploadForm');
        this.fileInput = document.getElementById('file');
        this.uploadButton = document.getElementById('uploadButton');
        this.progressContainer = document.getElementById('progressContainer');
        this.progressBar = document.getElementById('progressBar');
        this.progressText = document.getElementById('progressText');
        this.resultContainer = document.getElementById('resultContainer');
        this.resultContent = document.getElementById('resultContent');
        this.errorContainer = document.getElementById('errorContainer');
        this.errorMessage = document.getElementById('errorMessage');

        this.initEventListeners();
    }

    initEventListeners() {
        this.uploadForm.addEventListener('submit', this.handleUpload.bind(this));
    }

    async handleUpload(event) {
        event.preventDefault();

        const files = Array.from(this.fileInput.files);
        if (files.length === 0) {
            this.showError('Please select at least one file');
            return;
        }

        const invalid = files.filter(f => !f.type.startsWith('image/'));
        if (invalid.length > 0) {
            this.showError(`Not a valid image: ${invalid.map(f => f.name).join(', ')}`);
            return;
        }

        this.hideError();
        this.showProgress(true);
        this.resetProgress();

        try {
            await this.uploadAndProcess(files);
        } catch (error) {
            this.showError(error.message || 'Upload failed');
        } finally {
            this.showProgress(false);
        }
    }

    async uploadAndProcess(files) {
        const formData = new FormData();
        for (const file of files) {
            formData.append('file', file);
        }

        // Simulated progress: jump to 10% immediately, then tick up by 2–5% every
        // 500 ms while the fetch is in flight, capped at 85% so it never self-completes.
        this.updateProgress(10);
        let current = 10;
        const ticker = setInterval(() => {
            current = Math.min(85, current + 2 + Math.random() * 3);
            this.updateProgress(current);
        }, 500);

        let response;
        try {
            const csrfToken = document.querySelector('meta[name="csrf-token"]').content;
            response = await fetch('/upload', {
                method: 'POST',
                body: formData,
                headers: {
                    'X-Requested-With': 'XMLHttpRequest',
                    'X-CSRFToken': csrfToken,
                },
            });
        } finally {
            clearInterval(ticker);
            this.updateProgress(100);
        }

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        this.displayResults(data);
    }

    showProgress(show) {
        if (show) {
            this.progressContainer.classList.remove('hidden');
            this.resultContainer.classList.add('hidden');
            this.hideError();
        } else {
            this.progressContainer.classList.add('hidden');
        }
    }

    resetProgress() {
        this.progressBar.style.width = '0%';
        this.progressText.textContent = 'Processing: 0%';
    }

    updateProgress(percent) {
        this.progressBar.style.width = `${percent}%`;
        this.progressText.textContent = `Processing: ${Math.round(percent)}%`;
    }

    displayResults(data) {
        this.showProgress(false);
        this.resultContainer.classList.remove('hidden');

        if (data.error) {
            this.resultContent.innerHTML = `
                <div class="border border-red-400 bg-red-50 text-red-700 px-4 py-3 rounded">
                    <div class="font-semibold mb-1">Upload Error</div>
                    <div class="text-sm">${data.error}</div>
                </div>`;
            return;
        }

        const results = data.results || [];
        if (results.length === 0) {
            this.resultContent.innerHTML = '<p class="text-gray-500 text-sm">No results returned.</p>';
            return;
        }

        this.resultContent.innerHTML = results.map(r => this._renderImageCard(r)).join('');
    }

    _renderImageCard(data) {
        if (data.status !== 'success') {
            return `
                <div class="border border-red-300 bg-red-50 rounded-lg p-4 mb-4">
                    <div class="font-semibold text-red-700 mb-1">${this._esc(data.filename || 'Unknown file')}</div>
                    <div class="text-sm text-red-600">${this._esc(data.error || 'Processing failed')}</div>
                </div>`;
        }

        const isStub = data.is_stub === true;
        const headerCls = isStub ? 'bg-yellow-50 border-yellow-200' : 'bg-green-50 border-green-200';
        const titleCls  = isStub ? 'text-yellow-800' : 'text-green-800';

        let detectionsHtml = '';
        (data.detections || []).forEach((detection, index) => {
            const detId = detection.detection_id || '';
            detectionsHtml += `
                <div class="p-2 bg-white rounded border mb-2">
                    <div class="flex items-center">
                        <div class="w-8 h-8 bg-blue-100 text-blue-800 rounded-full flex items-center justify-center mr-3 font-semibold flex-shrink-0">
                            ${index + 1}
                        </div>
                        <div class="flex-1">
                            <div class="font-medium">Box: [${detection.x1.toFixed(1)}, ${detection.y1.toFixed(1)}] to [${detection.x2.toFixed(1)}, ${detection.y2.toFixed(1)}]</div>
                            <div class="text-sm text-gray-600">Confidence: ${(detection.confidence * 100).toFixed(1)}%</div>
                        </div>
                    </div>
                    ${detId ? `
                    <div class="mt-2 pl-11">
                        <label class="text-xs text-gray-500 font-medium">Container Code (ISO 6346)</label>
                        ${detection.container_code ? `
                        <div class="flex items-center gap-2 mt-1 mb-1">
                            <span class="text-xs font-medium text-gray-500">OCR read:</span>
                            <span class="font-mono font-semibold text-sm">${this._esc(detection.container_code)}</span>
                            <span class="text-xs px-1.5 py-0.5 rounded ${detection.is_valid ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'}">
                                ${detection.is_valid ? '✓ Valid' : '⚠ Unverified'}
                            </span>
                            ${detection.is_recovery_based ? `
                            <span class="text-xs px-1.5 py-0.5 rounded bg-orange-100 text-orange-700">
                                Recovery-based match — please verify
                            </span>` : ''}
                        </div>` : '<p class="text-xs text-gray-400 mt-1 mb-1">OCR could not read code — enter manually</p>'}
                        <div class="flex items-center gap-2 mt-1">
                            <input type="text" id="code-input-${detId}" maxlength="11"
                                placeholder="e.g. MSCU1234560"
                                value="${this._esc(detection.container_code || '')}"
                                class="border border-gray-300 rounded px-2 py-1 text-sm w-36 uppercase"
                                oninput="validateISO6346(this, 'code-status-${detId}')">
                            <span id="code-status-${detId}" class="text-xs"></span>
                            <button type="button" id="code-btn-${detId}"
                                onclick="saveContainerCode(${detId}, 'code-input-${detId}', 'code-save-msg-${detId}')"
                                class="text-xs bg-blue-500 text-white px-2 py-1 rounded hover:bg-blue-600">Save</button>
                            <span id="code-save-msg-${detId}" class="text-xs"></span>
                        </div>
                    </div>` : ''}
                </div>`;
        });

        let annotatedHtml = '';
        if (data.annotated_image_url) {
            annotatedHtml = `
                <div class="mt-4">
                    <p class="text-sm text-gray-600 mb-2 font-medium">Annotated Image:</p>
                    <img src="${data.annotated_image_url}" alt="Annotated detection result"
                        class="max-w-full rounded border border-gray-200" style="max-height:500px;">
                </div>`;
        }

        return `
            <div class="border rounded-lg p-4 mb-4 ${headerCls}">
                <div class="flex items-center justify-between mb-3">
                    <h4 class="font-semibold ${titleCls}">
                        ${this._esc(data.filename || '')}
                        &nbsp;·&nbsp;
                        ✓ Analysis Complete
                    </h4>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-3">
                    <div>
                        <p class="text-sm text-gray-600 mb-1">Model Used:</p>
                        <p class="font-medium ${titleCls}">${data.model_type}</p>
                    </div>
                    <div>
                        <p class="text-sm text-gray-600 mb-1">Processing Time:</p>
                        <p class="font-medium">${(data.processing_time * 1000).toFixed(0)}ms</p>
                    </div>
                    <div class="md:col-span-2">
                        <p class="text-sm text-gray-600 mb-2">Detections:</p>
                        <div class="space-y-2">${detectionsHtml}</div>
                    </div>
                </div>

                ${annotatedHtml}
            </div>`;
    }

    _esc(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    displayError(data) {
        this.resultContainer.classList.remove('hidden');
        this.resultContent.innerHTML = `
            <div class="border border-red-400 bg-red-50 text-red-700 px-4 py-3 rounded">
                <div class="font-semibold mb-2">Analysis Failed</div>
                <div class="text-sm">Model: ${data.model_type || 'unknown'}</div>
                <div class="text-sm">Error: ${data.error || 'Unknown error occurred'}</div>
            </div>`;
    }

    showError(message) {
        this.errorContainer.classList.remove('hidden');
        this.errorMessage.textContent = message;
        this.resultContainer.classList.add('hidden');
    }

    hideError() {
        this.errorContainer.classList.add('hidden');
    }
}


function _iso6346CharValue(c) {
    if (c >= '0' && c <= '9') return parseInt(c);
    const idx = c.charCodeAt(0) - 65; // 0 for A
    let v = idx + 10;
    if (v >= 11) v++;
    if (v >= 22) v++;
    if (v >= 33) v++;
    return v;
}

function validateISO6346(inputEl, statusId) {
    const raw = inputEl.value.replace(/[\s-]/g, '').toUpperCase();
    inputEl.value = raw;
    const statusEl = document.getElementById(statusId);
    const btnEl = document.getElementById(statusId.replace('code-status-', 'code-btn-'));

    const set = (msg, color, enable, borderColor) => {
        if (statusEl) { statusEl.textContent = msg; statusEl.style.color = color; }
        if (btnEl) btnEl.disabled = !enable;
        inputEl.style.borderColor = borderColor;
        inputEl.style.borderWidth = borderColor ? '2px' : '';
    };

    if (raw.length === 0) { set('', '', false, ''); return; }
    if (!/^[A-Z]{4}\d{7}$/.test(raw)) {
        set('✗ Invalid format (need 4 letters + 6 digits + check digit)', '#dc2626', false, '#dc2626');
        return;
    }

    let sum = 0;
    for (let i = 0; i < 10; i++) sum += _iso6346CharValue(raw[i]) * Math.pow(2, i);
    const expected = sum % 11 === 10 ? 0 : sum % 11;
    const actual = parseInt(raw[10]);
    if (actual === expected) {
        set('✓ Valid ISO 6346', '#16a34a', true, '#16a34a');
    } else {
        set(`✗ Check digit mismatch (expected: ${expected})`, '#dc2626', false, '#dc2626');
    }
}

async function saveContainerCode(detectionId, inputId, msgId) {
    const code = document.getElementById(inputId).value.trim().toUpperCase();
    const msgEl = document.getElementById(msgId);
    const csrfToken = document.querySelector('meta[name="csrf-token"]').content;
    try {
        const r = await fetch(`/detections/${detectionId}/code`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrfToken },
            body: JSON.stringify({ container_code: code }),
        });
        const d = await r.json();
        const success = !d.error && r.ok;
        if (msgEl) {
            msgEl.textContent = success ? 'Saved' : 'Save failed';
            msgEl.style.color = success ? '#16a34a' : '#dc2626';
            setTimeout(() => { if (msgEl) msgEl.textContent = ''; }, 2000);
        }
    } catch (e) {
        if (msgEl) {
            msgEl.textContent = 'Save failed';
            msgEl.style.color = '#dc2626';
            setTimeout(() => { if (msgEl) msgEl.textContent = ''; }, 2000);
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new InferenceUploader();
});

// Expose to global scope — required because type="module" creates its own scope,
// but inline onclick/oninput attributes resolve against window.
window.validateISO6346 = validateISO6346;
window.saveContainerCode = saveContainerCode;
