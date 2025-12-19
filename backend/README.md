# Nesting Tool API Backend

FastAPI backend for the Nesting Tool application.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements-backend.txt
```

### 2. Run the API Server

```bash
# From the project root directory:
uvicorn backend.main:app --reload --port 8000

# Or from the backend directory:
cd backend
python main.py
```

### 3. Access API Documentation

Once running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## API Endpoints

### Parts Management

- `POST /api/parts` - Create a new part
- `GET /api/parts` - Get all parts
- `GET /api/parts/{id}` - Get specific part
- `PUT /api/parts/{id}` - Update a part
- `DELETE /api/parts/{id}` - Delete a part
- `DELETE /api/parts` - Clear all parts

### Nesting Operations

- `POST /api/nest` - Nest provided parts
- `POST /api/nest/current` - Nest all parts in database

### Projects

- `POST /api/projects` - Save current parts as project
- `GET /api/projects` - List all projects
- `GET /api/projects/{id}` - Load a project

## Testing

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Create a part
curl -X POST "http://localhost:8000/api/parts" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "part-1",
    "label": "Test Part",
    "qty": 1,
    "shape_type": "rect",
    "width": 50,
    "height": 30,
    "allow_rotation": true,
    "meta": {}
  }'

# Get all parts
curl http://localhost:8000/api/parts

# Run nesting
curl -X POST "http://localhost:8000/api/nest" \
  -H "Content-Type: application/json" \
  -d '{
    "parts": [{
      "id": "part-1",
      "label": "Test Part",
      "qty": 1,
      "shape_type": "rect",
      "width": 50,
      "height": 30,
      "allow_rotation": true,
      "meta": {}
    }],
    "config": {
      "sheet_w": 139,
      "sheet_h": 80,
      "clearance": 0.25,
      "allow_rotation": true,
      "autosplit_rects": false
    }
  }'
```

### Using the Interactive Docs

1. Navigate to http://localhost:8000/docs
2. Click on any endpoint to expand it
3. Click "Try it out"
4. Fill in the parameters
5. Click "Execute"
6. See the response below

## Architecture

```
backend/
├── main.py           # FastAPI application and endpoints
├── models.py         # Pydantic data models
├── nesting.py        # Nesting algorithms
└── routers/          # Future: organized route modules
```

## Development

### Adding New Endpoints

1. Add route function to `main.py`
2. Define request/response models in `models.py`
3. Implement logic in appropriate module
4. Test using `/docs` interface

### Adding Database (Phase 4)

Uncomment database dependencies in `requirements-backend.txt` and add:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Configure database
engine = create_engine("sqlite:///./nesting.db")
SessionLocal = sessionmaker(bind=engine)
```

## Deployment

See main project documentation for deployment options:
- Railway
- Render
- DigitalOcean
- Docker/Docker Compose
