# Nesting Tool Pro - Web UI

Professional drag-and-drop interface for the Nesting Tool application.

## Features

✨ **Interactive Canvas**
- Drag-and-drop parts onto the canvas
- Visual representation with Fabric.js
- Grid snapping for precise placement
- Real-time mouse tracking

🎨 **Modern UI**
- Professional 3-panel layout
- Responsive design
- Smooth animations
- Loading states

🔧 **Parts Management**
- Quick add form for rectangles
- Parts library with visual cards
- One-click part deletion
- Click to add to canvas

✏️ **Constrained Drawing Tool** (NEW!)
- CAD-like width-locked drawing tool for custom countertop shapes
- Draw with fixed width (default 25.5", adjustable)
- Right-click to add 90° perpendicular corners automatically
- Left-click to finish drawing
- Real-time dimension labels on all sides
- Click edges to toggle polished/unfinished profiles
- Visual edge markers (green = polished, red dashed = unfinished)
- Escape key to cancel drawing in progress

🧩 **Nesting Integration**
- Connects to FastAPI backend
- Visual nesting results
- Utilization metrics (displayed in square feet)
- Multi-sheet visualization

⌨️ **Keyboard Shortcuts**
- `Ctrl/Cmd + N` - Run nesting
- `Ctrl/Cmd + A` - Focus add part form
- `Enter` - Submit add part form
- `Escape` - Cancel drawing in progress

## Quick Start

### Option 1: Open Directly (Simple Server Needed)

Since this is a static web app that makes API calls, you need to serve it through a web server (not just open the HTML file directly due to CORS).

#### **Using Python (Recommended)**

```bash
# Navigate to the frontend directory
cd frontend

# Start a simple HTTP server
python -m http.server 8080

# Open in browser:
# http://localhost:8080
```

#### **Using Node.js**

```bash
# Install http-server globally (one-time)
npm install -g http-server

# Navigate to frontend directory
cd frontend

# Start server
http-server -p 8080

# Open in browser:
# http://localhost:8080
```

#### **Using Live Server (VS Code)**

1. Install "Live Server" extension in VS Code
2. Right-click `index.html`
3. Select "Open with Live Server"

### Option 2: Integration with Backend (Production)

For production deployment, the frontend can be served by the FastAPI backend or deployed separately.

## Prerequisites

Before using the web UI:

1. **Backend API must be running**
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```

2. **API URL Configuration**
   - Default: `http://localhost:8000/api`
   - Change in `app.js` line 11 if needed

## Usage Workflow

### 1. Start the Backend
```bash
# In one terminal:
uvicorn backend.main:app --reload --port 8000
```

### 2. Start the Frontend
```bash
# In another terminal:
cd frontend
python -m http.server 8080
```

### 3. Open in Browser
Navigate to: `http://localhost:8080`

### 4. Use the Application

**Add Parts:**
1. Fill out the "Quick Add" form in the left sidebar
2. Click "Add Part"
3. Part appears in the Parts Library

**Arrange on Canvas:**
1. Click a part in the library to add to canvas
2. Drag parts around the canvas
3. Parts snap to grid for precision

**Run Nesting:**
1. Click "Nest Parts" button in toolbar
2. View results in right sidebar
3. See visual layout on canvas

**Use Constrained Drawing Tool:**
1. Click the pencil-ruler icon in the toolbar to activate Draw Mode
2. Adjust width in the "Width" input (default: 25.5")
3. **Left-click** on the canvas to start drawing
4. **Right-click** to add a 90° perpendicular corner at that point
5. Continue right-clicking to add more corners (automatically perpendicular)
6. **Left-click** again to finish and save the part
7. Press **Escape** to cancel drawing in progress

**Edit Edge Profiles:**
- Click on any edge of a drawn part to toggle between:
  - **Polished** (green solid line)
  - **Unfinished** (red dashed line)
- This helps track which edges need finishing/polishing

**Dimension Labels:**
- All dimensions are displayed in **inches** on each side during drawing
- Dimension labels remain visible after drawing is complete

## Architecture

```
frontend/
├── index.html     # Main HTML structure
├── styles.css     # Professional CSS styling
├── app.js         # JavaScript logic & API integration
└── README.md      # This file
```

### Key Technologies

- **Fabric.js** - Canvas manipulation and drag-and-drop
- **Font Awesome** - Icons
- **Vanilla JavaScript** - No framework dependencies
- **CSS Grid** - Modern responsive layout

## Configuration

### API Endpoint

Edit `app.js` line 11:
```javascript
const API_URL = 'http://localhost:8000/api';
```

### Canvas Settings

Edit in `app.js` `initCanvas()` function:
```javascript
canvas = new fabric.Canvas('fabricCanvas', {
    width: 1200,    // Canvas width
    height: 800,    // Canvas height
    backgroundColor: '#ffffff',
    // ... other options
});
```

### Grid Size

Edit `snapToGrid()` function:
```javascript
function snapToGrid(obj, gridSize = 10) {  // Change grid size here
    // ...
}
```

## Troubleshooting

### ❌ "Failed to connect to API"
- **Solution**: Make sure backend is running on port 8000
- Check: `http://localhost:8000/health` in browser

### ❌ CORS Errors
- **Solution**: Frontend must be served through HTTP server (not file://)
- Use Python server or Live Server extension

### ❌ Parts not appearing
- **Solution**: Check browser console for errors
- Verify API connection status (top right of UI)

### ❌ Canvas not rendering
- **Solution**: Check that Fabric.js CDN is accessible
- Try refreshing the page

## Browser Compatibility

✅ Chrome 90+
✅ Firefox 88+
✅ Safari 14+
✅ Edge 90+

## Development

### Adding New Features

1. **New UI Elements**: Edit `index.html`
2. **Styling**: Edit `styles.css`
3. **Logic**: Edit `app.js`

### Debugging

Open browser Developer Tools (F12):
- **Console**: View logs and errors
- **Network**: Monitor API calls
- **Elements**: Inspect DOM

### Code Structure

**app.js** is organized into sections:
- Initialization
- API Communication
- Parts Management
- Canvas Operations
- Nesting Operations
- Utility Functions

## Next Steps

- Add angle editing for drawn parts (adjust corners after drawing)
- Implement project save/load from UI
- Add export to PDF/DXF
- Mobile responsive improvements
- Implement undo/redo for canvas
- Add part editing functionality (modify existing parts)

## Support

For issues or questions:
1. Check browser console for errors
2. Verify API is running: `http://localhost:8000/docs`
3. See main project README for more help
