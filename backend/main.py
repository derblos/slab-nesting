"""
FastAPI Backend for Nesting Tool

REST API providing endpoints for parts management and nesting algorithms.
Run with: uvicorn backend.main:app --reload
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
import uuid

from .models import (
    Part,
    NestingConfig,
    NestingRequest,
    NestingResult,
    ErrorResponse
)
from .nesting import rectpack_nest, calculate_total_area

# Initialize FastAPI app
app = FastAPI(
    title="Nesting Tool API",
    description="REST API for rectangular and polygon nesting operations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (will be replaced with database in Phase 4)
parts_database: List[Part] = []
projects_database: dict = {}


@app.get("/", tags=["Root"])
async def root():
    """API root endpoint"""
    return {
        "message": "Nesting Tool API v1.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "parts_count": len(parts_database),
        "projects_count": len(projects_database)
    }


# ============================================================================
# PARTS ENDPOINTS
# ============================================================================

@app.post("/api/parts", response_model=Part, tags=["Parts"], status_code=status.HTTP_201_CREATED)
async def create_part(part: Part):
    """
    Create a new part

    - **id**: Unique identifier for the part
    - **label**: Display name for the part
    - **qty**: Quantity to nest (must be >= 1)
    - **shape_type**: Either 'rect' or 'polygon'
    - **width/height**: Dimensions for rectangular parts
    - **points**: Vertex coordinates for polygon parts
    - **allow_rotation**: Whether 0/90° rotation is allowed
    """
    # Check for duplicate ID
    if any(p.id == part.id for p in parts_database):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Part with ID '{part.id}' already exists"
        )

    parts_database.append(part)
    return part


@app.get("/api/parts", response_model=List[Part], tags=["Parts"])
async def get_parts():
    """Get all parts"""
    return parts_database


@app.get("/api/parts/{part_id}", response_model=Part, tags=["Parts"])
async def get_part(part_id: str):
    """Get a specific part by ID"""
    part = next((p for p in parts_database if p.id == part_id), None)
    if not part:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Part '{part_id}' not found"
        )
    return part


@app.put("/api/parts/{part_id}", response_model=Part, tags=["Parts"])
async def update_part(part_id: str, updated_part: Part):
    """Update an existing part"""
    for i, p in enumerate(parts_database):
        if p.id == part_id:
            parts_database[i] = updated_part
            return updated_part

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Part '{part_id}' not found"
    )


@app.delete("/api/parts/{part_id}", tags=["Parts"])
async def delete_part(part_id: str):
    """Delete a part"""
    global parts_database
    initial_length = len(parts_database)
    parts_database = [p for p in parts_database if p.id != part_id]

    if len(parts_database) == initial_length:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Part '{part_id}' not found"
        )

    return {"message": f"Part '{part_id}' deleted successfully"}


@app.delete("/api/parts", tags=["Parts"])
async def clear_all_parts():
    """Clear all parts from the database"""
    global parts_database
    count = len(parts_database)
    parts_database = []
    return {"message": f"Cleared {count} parts"}


# ============================================================================
# NESTING ENDPOINTS
# ============================================================================

@app.post("/api/nest", response_model=NestingResult, tags=["Nesting"])
async def nest_parts(request: NestingRequest):
    """
    Run the nesting algorithm

    Uses the rectpack algorithm for fast rectangular nesting.
    Supports auto-splitting of oversized parts and L-shape decomposition.

    Returns sheets with part placements and utilization statistics.
    """
    try:
        # Validate we have parts to nest
        if not request.parts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No parts provided for nesting"
            )

        # Run nesting algorithm
        sheets, utilization = rectpack_nest(request.parts, request.config)

        # Calculate areas
        used_area, total_area = calculate_total_area(sheets, request.config)

        return NestingResult(
            success=True,
            sheets=sheets,
            utilization=utilization,
            num_sheets=len(sheets),
            total_area_used=used_area,
            total_area_available=total_area,
            message=f"Successfully nested {len(request.parts)} parts onto {len(sheets)} sheet(s)"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Nesting failed: {str(e)}"
        )


@app.post("/api/nest/current", response_model=NestingResult, tags=["Nesting"])
async def nest_current_parts(config: NestingConfig):
    """
    Nest all parts currently in the database

    Convenience endpoint that nests all parts without requiring
    them to be sent in the request body.
    """
    if not parts_database:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No parts in database to nest"
        )

    request = NestingRequest(parts=parts_database, config=config)
    return await nest_parts(request)


# ============================================================================
# PROJECT ENDPOINTS
# ============================================================================

@app.post("/api/projects", tags=["Projects"])
async def save_project(name: str):
    """Save current parts as a project"""
    project_id = str(uuid.uuid4())
    projects_database[project_id] = {
        "id": project_id,
        "name": name,
        "parts": [p.model_dump() for p in parts_database]
    }
    return {"id": project_id, "message": f"Project '{name}' saved"}


@app.get("/api/projects", tags=["Projects"])
async def list_projects():
    """List all saved projects"""
    return list(projects_database.values())


@app.get("/api/projects/{project_id}", tags=["Projects"])
async def load_project(project_id: str):
    """Load a project by ID"""
    if project_id not in projects_database:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_id}' not found"
        )
    return projects_database[project_id]


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc)
        ).model_dump()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
