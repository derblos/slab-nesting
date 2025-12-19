/**
 * Nesting Tool Pro - Web UI JavaScript
 * Handles canvas interactions, API calls, and drag-and-drop functionality
 */

// Configuration
const API_URL = 'http://localhost:8000/api';
let canvas;
let parts = [];
let nestingResults = null;

// =================================================================
// Initialization
// =================================================================

window.addEventListener('load', () => {
    initCanvas();
    checkAPIConnection();
    loadParts();
    setupEventListeners();
});

function initCanvas() {
    const canvasEl = document.getElementById('fabricCanvas');
    const wrapper = canvasEl.parentElement;

    canvas = new fabric.Canvas('fabricCanvas', {
        width: wrapper.clientWidth - 40,
        height: wrapper.clientHeight - 40,
        backgroundColor: '#ffffff',
        selection: true,
        preserveObjectStacking: true
    });

    // Draw grid
    drawGrid();

    // Mouse tracking
    canvas.on('mouse:move', (e) => {
        const pointer = canvas.getPointer(e.e);
        document.getElementById('mousePosition').textContent =
            `X: ${Math.round(pointer.x)}, Y: ${Math.round(pointer.y)}`;
    });

    // Object events
    canvas.on('object:moving', (e) => {
        const obj = e.target;
        snapToGrid(obj);
    });

    canvas.on('object:modified', () => {
        updateStatus('Part moved');
    });

    // Handle window resize
    window.addEventListener('resize', () => {
        canvas.setDimensions({
            width: wrapper.clientWidth - 40,
            height: wrapper.clientHeight - 40
        });
        canvas.renderAll();
    });
}

function drawGrid() {
    const gridSize = 20;
    const width = canvas.width;
    const height = canvas.height;

    // Vertical lines
    for (let i = 0; i < width / gridSize; i++) {
        canvas.add(new fabric.Line([i * gridSize, 0, i * gridSize, height], {
            stroke: '#e0e0e0',
            strokeWidth: 1,
            selectable: false,
            evented: false,
            type: 'grid'
        }));
    }

    // Horizontal lines
    for (let i = 0; i < height / gridSize; i++) {
        canvas.add(new fabric.Line([0, i * gridSize, width, i * gridSize], {
            stroke: '#e0e0e0',
            strokeWidth: 1,
            selectable: false,
            evented: false,
            type: 'grid'
        }));
    }
}

function snapToGrid(obj, gridSize = 10) {
    obj.set({
        left: Math.round(obj.left / gridSize) * gridSize,
        top: Math.round(obj.top / gridSize) * gridSize
    });
}

function resetZoom() {
    canvas.setViewportTransform([1, 0, 0, 1, 0, 0]);
    canvas.renderAll();
    updateStatus('View reset');
}

// =================================================================
// API Communication
// =================================================================

async function checkAPIConnection() {
    const statusEl = document.getElementById('connectionStatus');

    try {
        const response = await fetch(`${API_URL.replace('/api', '')}/health`);
        const data = await response.json();

        if (data.status === 'healthy') {
            statusEl.classList.add('connected');
            statusEl.classList.remove('disconnected');
            statusEl.querySelector('span').textContent = 'Connected';
            updateStatus('Connected to API');
        }
    } catch (error) {
        statusEl.classList.add('disconnected');
        statusEl.classList.remove('connected');
        statusEl.querySelector('span').textContent = 'Disconnected';
        updateStatus('Failed to connect to API', 'error');
        console.error('API connection error:', error);
    }
}

async function apiRequest(endpoint, method = 'GET', data = null) {
    const options = {
        method,
        headers: {
            'Content-Type': 'application/json',
        }
    };

    if (data) {
        options.body = JSON.stringify(data);
    }

    try {
        const response = await fetch(`${API_URL}${endpoint}`, options);
        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'API request failed');
        }

        return result;
    } catch (error) {
        console.error('API Error:', error);
        updateStatus(`Error: ${error.message}`, 'error');
        throw error;
    }
}

// =================================================================
// Parts Management
// =================================================================

async function loadParts() {
    try {
        parts = await apiRequest('/parts');
        renderPartsList();
        updateStatus(`Loaded ${parts.length} part(s)`);
    } catch (error) {
        console.error('Failed to load parts:', error);
    }
}

async function refreshParts() {
    showLoading(true);
    await loadParts();
    showLoading(false);
}

function renderPartsList() {
    const listEl = document.getElementById('partsList');
    const countEl = document.getElementById('partsCount');

    countEl.textContent = parts.length;

    if (parts.length === 0) {
        listEl.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-inbox"></i>
                <p>No parts yet</p>
                <small>Add a part to get started</small>
            </div>
        `;
        return;
    }

    listEl.innerHTML = parts.map(part => `
        <div class="part-item" draggable="true" data-id="${part.id}">
            <div class="part-item-header">
                <div class="part-item-label">
                    <i class="fas fa-cube"></i> ${part.label}
                </div>
                <div class="part-item-actions">
                    <button class="btn btn-sm btn-icon" onclick="deletePart('${part.id}')" title="Delete">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
            <div class="part-item-info">
                <span><i class="fas fa-arrows-alt-h"></i> ${part.width}"</span>
                <span><i class="fas fa-arrows-alt-v"></i> ${part.height}"</span>
                <span><i class="fas fa-copy"></i> Qty: ${part.qty}</span>
                <span><i class="fas fa-sync-alt"></i> ${part.allow_rotation ? 'Yes' : 'No'}</span>
            </div>
        </div>
    `).join('');

    // Setup drag and drop
    document.querySelectorAll('.part-item').forEach(el => {
        el.addEventListener('dragstart', handleDragStart);
        el.addEventListener('click', () => addPartToCanvas(parts.find(p => p.id === el.dataset.id)));
    });
}

function handleDragStart(e) {
    const partId = e.target.closest('.part-item').dataset.id;
    e.dataTransfer.setData('partId', partId);
}

async function addPart() {
    const label = document.getElementById('partLabel').value;
    const width = parseFloat(document.getElementById('partWidth').value);
    const height = parseFloat(document.getElementById('partHeight').value);
    const qty = parseInt(document.getElementById('partQty').value);
    const allowRotation = document.getElementById('partRotation').checked;

    if (!label || !width || !height || !qty) {
        updateStatus('Please fill all fields', 'error');
        return;
    }

    const part = {
        id: `part-${Date.now()}`,
        label,
        qty,
        shape_type: 'rect',
        width,
        height,
        points: null,
        allow_rotation: allowRotation,
        meta: {}
    };

    try {
        showLoading(true);
        await apiRequest('/parts', 'POST', part);
        await loadParts();
        updateStatus(`Added ${label}`);

        // Auto-increment label
        const match = label.match(/(\d+)$/);
        if (match) {
            const num = parseInt(match[1]) + 1;
            document.getElementById('partLabel').value = label.replace(/\d+$/, num);
        }
    } catch (error) {
        // Error already handled in apiRequest
    } finally {
        showLoading(false);
    }
}

async function deletePart(partId) {
    if (!confirm('Delete this part?')) return;

    try {
        showLoading(true);
        await apiRequest(`/parts/${partId}`, 'DELETE');
        await loadParts();
        updateStatus('Part deleted');
    } catch (error) {
        // Error already handled
    } finally {
        showLoading(false);
    }
}

async function clearAllParts() {
    if (!confirm('Clear all parts? This cannot be undone.')) return;

    try {
        showLoading(true);
        await apiRequest('/parts', 'DELETE');
        await loadParts();
        canvas.clear();
        drawGrid();
        updateStatus('All parts cleared');
    } catch (error) {
        // Error already handled
    } finally {
        showLoading(false);
    }
}

// =================================================================
// Canvas Operations
// =================================================================

function addPartToCanvas(part, x = null, y = null) {
    if (!part) return;

    const scale = 3; // Scale factor for visibility
    const posX = x || Math.random() * (canvas.width - part.width * scale);
    const posY = y || Math.random() * (canvas.height - part.height * scale);

    const rect = new fabric.Rect({
        left: posX,
        top: posY,
        width: part.width * scale,
        height: part.height * scale,
        fill: 'rgba(52, 152, 219, 0.2)',
        stroke: '#3498db',
        strokeWidth: 2,
        cornerColor: '#3498db',
        cornerSize: 10,
        transparentCorners: false,
        partId: part.id,
        partData: part
    });

    const text = new fabric.Text(part.label, {
        left: posX + 10,
        top: posY + 10,
        fontSize: 14,
        fill: '#2c3e50',
        fontWeight: 'bold',
        selectable: false,
        evented: false
    });

    const group = new fabric.Group([rect, text], {
        partId: part.id,
        partData: part
    });

    canvas.add(group);
    canvas.setActiveObject(group);
    canvas.renderAll();

    updateStatus(`Added ${part.label} to canvas`);
}

// =================================================================
// Nesting Operations
// =================================================================

async function runNesting() {
    if (parts.length === 0) {
        updateStatus('Add parts first', 'error');
        return;
    }

    const config = {
        sheet_w: parseFloat(document.getElementById('sheetWidth').value),
        sheet_h: parseFloat(document.getElementById('sheetHeight').value),
        clearance: parseFloat(document.getElementById('clearance').value),
        allow_rotation: document.getElementById('allowRotation').checked,
        autosplit_rects: false,
        seam_gap: 0.125,
        min_leg: 6.0,
        prefer_long_split: true,
        enable_L_seams: true,
        L_max_leg: 48.0,
        grid_step: 0.5,
        units: 'in',
        precision: 2
    };

    const request = {
        parts: parts,
        config: config,
        mode: 'rectpack'
    };

    try {
        showLoading(true);
        updateStatus('Running nesting algorithm...');

        const result = await apiRequest('/nest', 'POST', request);
        nestingResults = result;

        displayNestingResults(result);
        visualizeNesting(result);

        updateStatus(`Nesting complete: ${result.num_sheets} sheet(s), ${(result.utilization * 100).toFixed(1)}% utilization`);
    } catch (error) {
        updateStatus('Nesting failed', 'error');
    } finally {
        showLoading(false);
    }
}

function displayNestingResults(result) {
    const resultsEl = document.getElementById('results');

    const html = `
        <div class="result-card">
            <h3><i class="fas fa-chart-bar"></i> Summary</h3>
            <div class="result-metric">
                <span class="result-metric-label">Sheets Used</span>
                <span class="result-metric-value">${result.num_sheets}</span>
            </div>
            <div class="result-metric">
                <span class="result-metric-label">Utilization</span>
                <span class="result-metric-value">${(result.utilization * 100).toFixed(1)}%</span>
            </div>
            <div class="result-metric">
                <span class="result-metric-label">Area Used</span>
                <span class="result-metric-value">${result.total_area_used.toFixed(2)} sq in</span>
            </div>
            <div class="result-metric">
                <span class="result-metric-label">Total Area</span>
                <span class="result-metric-value">${result.total_area_available.toFixed(2)} sq in</span>
            </div>
        </div>

        ${result.sheets.map((sheet, i) => `
            <div class="result-card">
                <h3><i class="fas fa-file"></i> Sheet ${i + 1}</h3>
                <div class="result-metric">
                    <span class="result-metric-label">Parts</span>
                    <span class="result-metric-value">${sheet.placements.length}</span>
                </div>
                <div style="font-size: 12px; margin-top: 10px; color: #7f8c8d;">
                    ${sheet.placements.map(p => `• ${p.label}`).join('<br>')}
                </div>
            </div>
        `).join('')}
    `;

    resultsEl.innerHTML = html;
}

function visualizeNesting(result) {
    // Clear canvas but keep grid
    const objects = canvas.getObjects();
    objects.forEach(obj => {
        if (obj.type !== 'grid' && obj.type !== 'line') {
            canvas.remove(obj);
        }
    });

    // Draw each sheet
    result.sheets.forEach((sheet, sheetIndex) => {
        const offsetX = sheetIndex * (sheet.sheet_w + 20) * 3;

        // Draw sheet boundary
        const sheetRect = new fabric.Rect({
            left: offsetX,
            top: 20,
            width: sheet.sheet_w * 3,
            height: sheet.sheet_h * 3,
            fill: 'transparent',
            stroke: '#2c3e50',
            strokeWidth: 3,
            selectable: false,
            evented: false
        });
        canvas.add(sheetRect);

        // Draw sheet label
        const sheetLabel = new fabric.Text(`Sheet ${sheetIndex + 1}`, {
            left: offsetX + 10,
            top: 5,
            fontSize: 16,
            fontWeight: 'bold',
            fill: '#2c3e50',
            selectable: false,
            evented: false
        });
        canvas.add(sheetLabel);

        // Draw parts on sheet
        sheet.placements.forEach(placement => {
            const partRect = new fabric.Rect({
                left: offsetX + placement.x * 3,
                top: 20 + placement.y * 3,
                width: placement.w * 3,
                height: placement.h * 3,
                fill: 'rgba(52, 152, 219, 0.3)',
                stroke: '#3498db',
                strokeWidth: 2,
                selectable: true,
                hasControls: false,
                lockMovementX: true,
                lockMovementY: true
            });

            const partLabel = new fabric.Text(placement.label, {
                left: offsetX + placement.x * 3 + 5,
                top: 20 + placement.y * 3 + 5,
                fontSize: 12,
                fill: '#2c3e50',
                fontWeight: 'bold',
                selectable: false,
                evented: false
            });

            canvas.add(partRect);
            canvas.add(partLabel);
        });
    });

    canvas.renderAll();
}

// =================================================================
// Utility Functions
// =================================================================

function updateStatus(message, type = 'info') {
    const statusEl = document.getElementById('statusMessage');
    statusEl.textContent = message;

    // Add visual feedback
    const icon = statusEl.previousElementSibling;
    icon.className = 'fas ' + (type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle');

    console.log(`[${type.toUpperCase()}] ${message}`);
}

function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    if (show) {
        overlay.classList.add('active');
    } else {
        overlay.classList.remove('active');
    }
}

function setupEventListeners() {
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey || e.metaKey) {
            if (e.key === 'n') {
                e.preventDefault();
                runNesting();
            } else if (e.key === 'a') {
                e.preventDefault();
                document.getElementById('partLabel').focus();
            }
        }
    });

    // Enter key to add part
    document.getElementById('partHeight').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            addPart();
        }
    });
}

// =================================================================
// Export for global access
// =================================================================

window.canvas = canvas;
window.addPart = addPart;
window.deletePart = deletePart;
window.clearAllParts = clearAllParts;
window.runNesting = runNesting;
window.loadParts = loadParts;
window.refreshParts = refreshParts;
window.resetZoom = resetZoom;
