/**
 * Nesting Tool Pro - Enhanced Web UI with Constrained Drawing
 * Features: Constrained drawing, edge profiles, dimension labels, angle editing
 */

// Configuration
const API_URL = 'http://localhost:8000/api';
let canvas;
let parts = [];
let nestingResults = null;

// Drawing Mode State
let drawingMode = false;
let currentDrawing = null;
let drawingPoints = [];
let drawingPreview = null;
let drawingLabels = [];

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
    canvas.on('mouse:move', handleMouseMove);
    canvas.on('mouse:down', handleMouseDown);

    // Object events
    canvas.on('object:moving', (e) => {
        snapToGrid(e.target);
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

    // Prevent context menu on canvas
    canvasEl.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        return false;
    });
}

function drawGrid() {
    const gridSize = 20;
    const width = canvas.width;
    const height = canvas.height;

    for (let i = 0; i < width / gridSize; i++) {
        canvas.add(new fabric.Line([i * gridSize, 0, i * gridSize, height], {
            stroke: '#e0e0e0',
            strokeWidth: 1,
            selectable: false,
            evented: false,
            objectType: 'grid'
        }));
    }

    for (let i = 0; i < height / gridSize; i++) {
        canvas.add(new fabric.Line([0, i * gridSize, width, i * gridSize], {
            stroke: '#e0e0e0',
            strokeWidth: 1,
            selectable: false,
            evented: false,
            objectType: 'grid'
        }));
    }
}

// =================================================================
// Drawing Mode
// =================================================================

function toggleDrawMode() {
    drawingMode = !drawingMode;
    const btn = document.getElementById('drawModeBtn');
    const controls = document.getElementById('drawControls');
    const wrapper = document.querySelector('.canvas-wrapper');
    const hint = document.getElementById('hintText');

    if (drawingMode) {
        btn.classList.add('active');
        controls.style.display = 'flex';
        controls.style.alignItems = 'center';
        controls.style.gap = '10px';
        wrapper.classList.add('drawing-mode');
        hint.textContent = 'Left-click to start | Right-click to add corner | Left-click to finish';

        // Disable object selection and interaction during drawing
        canvas.selection = false;
        canvas.defaultCursor = 'crosshair';
        canvas.hoverCursor = 'crosshair';
        canvas.forEachObject(obj => {
            obj.selectable = false;
            obj.evented = false;
        });

        console.log('✅ Draw mode ACTIVATED');
        updateStatus('Draw mode active - Left-click to start drawing');
    } else {
        btn.classList.remove('active');
        controls.style.display = 'none';
        wrapper.classList.remove('drawing-mode');
        hint.textContent = 'Click parts to add to canvas';

        // Re-enable object selection
        canvas.selection = true;
        canvas.defaultCursor = 'default';
        canvas.hoverCursor = 'move';
        canvas.forEachObject(obj => {
            if (obj.objectType !== 'grid' && obj.objectType !== 'preview') {
                obj.selectable = true;
                obj.evented = true;
            }
        });

        cancelDrawing();
        console.log('❌ Draw mode DEACTIVATED');
        updateStatus('Draw mode disabled');
    }
}

function handleMouseMove(e) {
    const pointer = canvas.getPointer(e.e);
    document.getElementById('mousePosition').textContent =
        `X: ${Math.round(pointer.x)}, Y: ${Math.round(pointer.y)}`;

    if (drawingMode && currentDrawing && drawingPoints.length > 0) {
        updateDrawingPreview(pointer);
    }
}

function handleMouseDown(e) {
    if (!drawingMode) return;

    // Get the raw event
    const evt = e.e;

    // Prevent default to stop context menu and other interference
    evt.preventDefault();
    evt.stopPropagation();

    const pointer = canvas.getPointer(evt);

    // Debug logging - check browser console (F12)
    console.log('Mouse down detected:', {
        button: evt.button,
        which: evt.which,
        buttons: evt.buttons,
        currentDrawing: currentDrawing,
        pointsCount: drawingPoints.length
    });

    // More robust button detection for Windows
    const isLeftClick = (evt.button === 0 && evt.buttons === 1) || (evt.which === 1);
    const isRightClick = (evt.button === 2 && evt.buttons === 2) || (evt.which === 3);

    console.log('Click type:', { isLeftClick, isRightClick });

    if (isLeftClick) {
        if (!currentDrawing) {
            // First left click - start drawing
            console.log('Starting drawing at:', pointer);
            startDrawing(pointer);
        } else {
            // Second left click - finish drawing (need at least 2 points)
            if (drawingPoints.length < 2) {
                console.log('Not enough points to finish');
                updateStatus('❌ Need at least one corner! Right-click to add corners, then left-click to finish');
                return;
            }
            console.log('Finishing drawing');
            finishDrawing();
        }
    }
    else if (isRightClick) {
        // Right click - add corner
        if (currentDrawing) {
            console.log('Adding corner at:', pointer);
            addCorner(pointer);
        } else {
            console.log('Not drawing yet, cannot add corner');
            updateStatus('Left-click first to start drawing');
        }
    }
    else {
        console.log('⚠️ Unknown button pressed');
    }
}

function startDrawing(point) {
    currentDrawing = true;
    drawingPoints = [{ x: point.x, y: point.y }];

    updateStatus('Drawing started - Right-click to add corners, Left-click to finish');
}

function addCorner(point) {
    if (!currentDrawing || drawingPoints.length === 0) return;

    const lastPoint = drawingPoints[drawingPoints.length - 1];
    const width = parseFloat(document.getElementById('drawWidth').value) * 3; // Scale factor

    // Calculate perpendicular direction (90 degrees from last segment)
    let direction;
    if (drawingPoints.length === 1) {
        // First segment - use mouse direction
        const dx = point.x - lastPoint.x;
        const dy = point.y - lastPoint.y;
        const angle = Math.atan2(dy, dx);
        const snappedAngle = Math.round(angle / (Math.PI / 2)) * (Math.PI / 2);
        direction = { x: Math.cos(snappedAngle), y: Math.sin(snappedAngle) };
    } else {
        // Subsequent segments - perpendicular to last
        const prevPoint = drawingPoints[drawingPoints.length - 2];
        const dx = lastPoint.x - prevPoint.x;
        const dy = lastPoint.y - prevPoint.y;
        // Rotate 90 degrees
        direction = { x: -dy, y: dx };
        const length = Math.sqrt(direction.x * direction.x + direction.y * direction.y);
        if (length > 0) {
            direction.x /= length;
            direction.y /= length;
        }
    }

    // Calculate length based on mouse position
    const dx = point.x - lastPoint.x;
    const dy = point.y - lastPoint.y;
    const length = Math.abs(dx * direction.x + dy * direction.y);

    // Add new point along the snapped direction
    const newPoint = {
        x: lastPoint.x + direction.x * length,
        y: lastPoint.y + direction.y * length
    };

    drawingPoints.push(newPoint);
    updateDrawingPreview(point);
    updateStatus(`Added corner (${drawingPoints.length} points) - Right-click for more corners, Left-click to finish`);
}

function updateDrawingPreview(currentPointer) {
    // Remove old preview
    if (drawingPreview) {
        canvas.remove(drawingPreview);
    }
    drawingLabels.forEach(label => canvas.remove(label));
    drawingLabels = [];

    if (drawingPoints.length === 0) return;

    const width = parseFloat(document.getElementById('drawWidth').value) * 3;
    const lastPoint = drawingPoints[drawingPoints.length - 1];

    // Calculate preview line direction
    let direction;
    if (drawingPoints.length === 1) {
        const dx = currentPointer.x - lastPoint.x;
        const dy = currentPointer.y - lastPoint.y;
        const angle = Math.atan2(dy, dx);
        const snappedAngle = Math.round(angle / (Math.PI / 2)) * (Math.PI / 2);
        direction = { x: Math.cos(snappedAngle), y: Math.sin(snappedAngle) };
    } else {
        const prevPoint = drawingPoints[drawingPoints.length - 2];
        const dx = lastPoint.x - prevPoint.x;
        const dy = lastPoint.y - prevPoint.y;
        direction = { x: -dy, y: dx };
        const length = Math.sqrt(direction.x * direction.x + direction.y * direction.y);
        if (length > 0) {
            direction.x /= length;
            direction.y /= length;
        }
    }

    const dx = currentPointer.x - lastPoint.x;
    const dy = currentPointer.y - lastPoint.y;
    const length = Math.abs(dx * direction.x + dy * direction.y);

    const previewEnd = {
        x: lastPoint.x + direction.x * length,
        y: lastPoint.y + direction.y * length
    };

    // Create centerline points including preview
    const allPoints = [...drawingPoints, previewEnd];

    // Expand the path to show full width rectangle
    const polygonPoints = expandPathWithWidth(allPoints, width);

    // Draw preview polygon with width
    drawingPreview = new fabric.Polygon(polygonPoints, {
        fill: 'rgba(52, 152, 219, 0.3)',  // Semi-transparent blue
        stroke: '#3498db',
        strokeWidth: 2,
        strokeDashArray: [8, 4],
        selectable: false,
        evented: false,
        objectType: 'preview'
    });
    canvas.add(drawingPreview);

    // Draw dimension labels on all segments
    for (let i = 0; i < allPoints.length - 1; i++) {
        const p1 = allPoints[i];
        const p2 = allPoints[i + 1];

        // Add dimension label
        const segLength = Math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2) / 3;
        const midX = (p1.x + p2.x) / 2;
        const midY = (p1.y + p2.y) / 2;

        const isPreview = (i === allPoints.length - 2);
        const label = new fabric.Text(`${segLength.toFixed(1)}"`, {
            left: midX,
            top: midY - 20,
            fontSize: 14,
            fill: 'white',
            backgroundColor: isPreview ? 'rgba(52, 152, 219, 0.9)' : 'rgba(44, 62, 80, 0.9)',
            padding: 6,
            selectable: false,
            evented: false,
            objectType: 'preview'
        });
        drawingLabels.push(label);
        canvas.add(label);
    }

    // Add width label
    const widthInches = width / 3;
    const widthLabel = new fabric.Text(`Width: ${widthInches.toFixed(1)}"`, {
        left: allPoints[0].x,
        top: allPoints[0].y - 40,
        fontSize: 14,
        fill: 'white',
        backgroundColor: 'rgba(231, 76, 60, 0.9)',
        padding: 6,
        selectable: false,
        evented: false,
        objectType: 'preview'
    });
    drawingLabels.push(widthLabel);
    canvas.add(widthLabel);

    canvas.renderAll();
}

async function finishDrawing() {
    if (!currentDrawing || drawingPoints.length < 2) {
        cancelDrawing();
        return;
    }

    // Create polygon from points
    const width = parseFloat(document.getElementById('drawWidth').value) * 3;

    // Convert points to actual polygon (with width)
    const expandedPoints = expandPathWithWidth(drawingPoints, width);

    // Clear preview
    canvas.getObjects().forEach(obj => {
        if (obj.objectType === 'preview') {
            canvas.remove(obj);
        }
    });

    // Create part
    const label = prompt('Enter part label:', `Part ${parts.length + 1}`);
    if (!label) {
        cancelDrawing();
        return;
    }

    // Calculate bounds
    const xs = expandedPoints.map(p => p.x);
    const ys = expandedPoints.map(p => p.y);
    const minX = Math.min(...xs);
    const minY = Math.min(...ys);
    const maxX = Math.max(...xs);
    const maxY = Math.max(...ys);

    const partWidth = (maxX - minX) / 3;
    const partHeight = (maxY - minY) / 3;

    // Create part polygon on canvas with edge profiles
    const polygon = new fabric.Polygon(expandedPoints, {
        left: 0,
        top: 0,
        fill: 'rgba(52, 152, 219, 0.2)',
        stroke: '#3498db',
        strokeWidth: 2,
        objectCaching: false,
        partLabel: label,
        edgeProfiles: new Array(expandedPoints.length).fill('unfinished') // All edges start unfinished
    });

    // Add dimension labels
    addDimensionLabels(polygon, drawingPoints);

    // Add edge profile markers
    addEdgeMarkers(polygon, expandedPoints);

    canvas.add(polygon);
    canvas.renderAll();

    // Save to backend
    try {
        showLoading(true);
        const part = {
            id: `part-${Date.now()}`,
            label: label,
            qty: 1,
            shape_type: 'polygon',
            width: partWidth,
            height: partHeight,
            points: expandedPoints.map(p => [p.x / 3, p.y / 3]),
            allow_rotation: true,
            meta: {
                drawingPoints: drawingPoints,
                width: parseFloat(document.getElementById('drawWidth').value)
            }
        };

        await apiRequest('/parts', 'POST', part);
        await loadParts();
        updateStatus(`Created ${label}`);
    } catch (error) {
        updateStatus('Failed to save part', 'error');
    } finally {
        showLoading(false);
    }

    // Reset drawing
    currentDrawing = null;
    drawingPoints = [];
    drawingPreview = null;
    drawingLabels = [];
}

function cancelDrawing() {
    currentDrawing = null;
    drawingPoints = [];

    if (drawingPreview) {
        canvas.remove(drawingPreview);
        drawingPreview = null;
    }

    drawingLabels.forEach(label => canvas.remove(label));
    drawingLabels = [];

    canvas.getObjects().forEach(obj => {
        if (obj.objectType === 'preview') {
            canvas.remove(obj);
        }
    });

    canvas.renderAll();
}

function expandPathWithWidth(centerPoints, width) {
    // Create a polygon by offsetting the path by width/2 on both sides
    const halfWidth = width / 2;
    const result = [];

    if (centerPoints.length < 2) return [];

    // Build one side of the path
    for (let i = 0; i < centerPoints.length - 1; i++) {
        const p1 = centerPoints[i];
        const p2 = centerPoints[i + 1];

        // Calculate perpendicular
        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const length = Math.sqrt(dx * dx + dy * dy);
        if (length === 0) continue;

        const perpX = -dy / length * halfWidth;
        const perpY = dx / length * halfWidth;

        if (i === 0) {
            result.push({ x: p1.x + perpX, y: p1.y + perpY });
        }
        result.push({ x: p2.x + perpX, y: p2.y + perpY });
    }

    // Add end cap for the last point
    const lastSegIdx = centerPoints.length - 2;
    const p1 = centerPoints[lastSegIdx];
    const p2 = centerPoints[lastSegIdx + 1];
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const length = Math.sqrt(dx * dx + dy * dy);
    if (length > 0) {
        const perpX = -dy / length * halfWidth;
        const perpY = dx / length * halfWidth;
        result.push({ x: p2.x - perpX, y: p2.y - perpY });
    }

    // Build opposite side (going backwards)
    for (let i = centerPoints.length - 2; i >= 0; i--) {
        const p1 = centerPoints[i];
        const p2 = centerPoints[i + 1];

        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const length = Math.sqrt(dx * dx + dy * dy);
        if (length === 0) continue;

        const perpX = -dy / length * halfWidth;
        const perpY = dx / length * halfWidth;

        if (i > 0) {
            result.push({ x: p1.x - perpX, y: p1.y - perpY });
        }
    }

    // Add start cap
    const firstP1 = centerPoints[0];
    const firstP2 = centerPoints[1];
    const firstDx = firstP2.x - firstP1.x;
    const firstDy = firstP2.y - firstP1.y;
    const firstLength = Math.sqrt(firstDx * firstDx + firstDy * firstDy);
    if (firstLength > 0) {
        const perpX = -firstDy / firstLength * halfWidth;
        const perpY = firstDx / firstLength * halfWidth;
        result.push({ x: firstP1.x - perpX, y: firstP1.y - perpY });
    }

    return result;
}

function addDimensionLabels(polygon, centerPoints) {
    for (let i = 0; i < centerPoints.length - 1; i++) {
        const p1 = centerPoints[i];
        const p2 = centerPoints[i + 1];

        const length = Math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2) / 3;
        const midX = (p1.x + p2.x) / 2;
        const midY = (p1.y + p2.y) / 2;

        const label = new fabric.Text(`${length.toFixed(2)}"`, {
            left: midX,
            top: midY - 15,
            fontSize: 12,
            fill: 'white',
            backgroundColor: 'rgba(44, 62, 80, 0.9)',
            padding: 4,
            selectable: false,
            evented: false,
            objectType: 'dimension'
        });
        canvas.add(label);
    }
}

function addEdgeMarkers(polygon, points) {
    // Add click handlers for edge profiles
    polygon.on('mousedown', function(e) {
        if (!drawingMode) {
            const pointer = canvas.getPointer(e.e);
            toggleEdgeProfile(polygon, pointer);
        }
    });
}

function toggleEdgeProfile(polygon, clickPoint) {
    // Find closest edge
    const points = polygon.points;
    let closestEdge = 0;
    let minDist = Infinity;

    for (let i = 0; i < points.length - 1; i++) {
        const p1 = points[i];
        const p2 = points[i + 1];
        const dist = distanceToSegment(clickPoint, p1, p2);
        if (dist < minDist) {
            minDist = dist;
            closestEdge = i;
        }
    }

    // Toggle edge profile
    if (!polygon.edgeProfiles) {
        polygon.edgeProfiles = new Array(points.length).fill('unfinished');
    }

    polygon.edgeProfiles[closestEdge] =
        polygon.edgeProfiles[closestEdge] === 'polished' ? 'unfinished' : 'polished';

    // Update visual
    updatePolygonEdges(polygon);
    canvas.renderAll();

    updateStatus(`Edge ${closestEdge + 1}: ${polygon.edgeProfiles[closestEdge]}`);
}

function distanceToSegment(point, p1, p2) {
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const length = Math.sqrt(dx * dx + dy * dy);

    if (length === 0) return Math.sqrt((point.x - p1.x) ** 2 + (point.y - p1.y) ** 2);

    const t = Math.max(0, Math.min(1, ((point.x - p1.x) * dx + (point.y - p1.y) * dy) / (length * length)));
    const projX = p1.x + t * dx;
    const projY = p1.y + t * dy;

    return Math.sqrt((point.x - projX) ** 2 + (point.y - projY) ** 2);
}

function updatePolygonEdges(polygon) {
    const points = polygon.points;

    // Update stroke based on edge profiles
    // This is simplified - in production you'd draw each edge separately
    const hasPolished = polygon.edgeProfiles && polygon.edgeProfiles.some(p => p === 'polished');
    const hasUnfinished = polygon.edgeProfiles && polygon.edgeProfiles.some(p => p === 'unfinished');

    if (hasPolished && !hasUnfinished) {
        polygon.set({ stroke: '#27ae60', strokeWidth: 4 });
    } else if (hasUnfinished && !hasPolished) {
        polygon.set({ stroke: '#e74c3c', strokeWidth: 4, strokeDashArray: [8, 4] });
    } else {
        polygon.set({ stroke: '#f39c12', strokeWidth: 4 });
    }
}

// =================================================================
// Standard Functions (Updated)
// =================================================================

function clearCanvas() {
    canvas.getObjects().forEach(obj => {
        if (obj.objectType !== 'grid') {
            canvas.remove(obj);
        }
    });
    canvas.renderAll();
    updateStatus('Canvas cleared');
}

function resetZoom() {
    canvas.setViewportTransform([1, 0, 0, 1, 0, 0]);
    canvas.renderAll();
    updateStatus('View reset');
}

function snapToGrid(obj, gridSize = 10) {
    obj.set({
        left: Math.round(obj.left / gridSize) * gridSize,
        top: Math.round(obj.top / gridSize) * gridSize
    });
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
                <span><i class="fas fa-arrows-alt-h"></i> ${part.width ? part.width.toFixed(2) : '—'}"</span>
                <span><i class="fas fa-arrows-alt-v"></i> ${part.height ? part.height.toFixed(2) : '—'}"</span>
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
        clearCanvas();
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

    const scale = 3;
    const posX = x || Math.random() * (canvas.width - part.width * scale);
    const posY = y || Math.random() * (canvas.height - part.height * scale);

    if (part.shape_type === 'rect') {
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

        canvas.add(rect);
        canvas.add(text);
    } else if (part.shape_type === 'polygon' && part.points) {
        const scaledPoints = part.points.map(p => ({ x: p[0] * scale, y: p[1] * scale }));
        const polygon = new fabric.Polygon(scaledPoints, {
            left: posX,
            top: posY,
            fill: 'rgba(52, 152, 219, 0.2)',
            stroke: '#3498db',
            strokeWidth: 2,
            partId: part.id,
            partData: part
        });
        canvas.add(polygon);
    }

    canvas.renderAll();
    updateStatus(`Added ${part.label} to canvas`);
}

// =================================================================
// Nesting Operations (Updated for Square Feet)
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

    // Convert to square feet (12" x 12" = 144 sq in per sq ft)
    const usedSqFt = (result.total_area_used / 144).toFixed(2);
    const totalSqFt = (result.total_area_available / 144).toFixed(2);

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
                <span class="result-metric-value">${usedSqFt} sq ft</span>
            </div>
            <div class="result-metric">
                <span class="result-metric-label">Total Area</span>
                <span class="result-metric-value">${totalSqFt} sq ft</span>
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
    canvas.getObjects().forEach(obj => {
        if (obj.objectType !== 'grid') {
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
            } else if (e.key === 'd') {
                e.preventDefault();
                toggleDrawMode();
            }
        }
        // Escape to cancel drawing
        if (e.key === 'Escape' && drawingMode) {
            cancelDrawing();
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
window.clearCanvas = clearCanvas;
window.toggleDrawMode = toggleDrawMode;
