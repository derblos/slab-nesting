"""
Data models for the Nesting Tool API

Pydantic models for request/response validation and serialization.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple


class Part(BaseModel):
    """Represents a part to be nested"""
    id: str
    label: str
    qty: int = Field(ge=1, description="Quantity must be at least 1")
    shape_type: str = Field(description="Either 'rect' or 'polygon'")
    width: Optional[float] = Field(None, ge=0)
    height: Optional[float] = Field(None, ge=0)
    points: Optional[List[Tuple[float, float]]] = None
    allow_rotation: bool = True
    meta: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "part-123",
                "label": "Countertop",
                "qty": 1,
                "shape_type": "rect",
                "width": 134.75,
                "height": 53.5,
                "points": None,
                "allow_rotation": True,
                "meta": {}
            }
        }


class NestingConfig(BaseModel):
    """Configuration for nesting operations"""
    sheet_w: float = Field(gt=0, description="Sheet width (must be positive)")
    sheet_h: float = Field(gt=0, description="Sheet height (must be positive)")
    clearance: float = Field(ge=0, description="Clearance between parts")
    allow_rotation: bool = True
    autosplit_rects: bool = True
    seam_gap: float = 0.125
    min_leg: float = 6.0
    prefer_long_split: bool = True
    enable_L_seams: bool = True
    L_max_leg: float = 48.0
    grid_step: float = 0.5
    units: str = "in"
    precision: int = 2

    class Config:
        json_schema_extra = {
            "example": {
                "sheet_w": 139.0,
                "sheet_h": 80.0,
                "clearance": 0.25,
                "allow_rotation": True,
                "autosplit_rects": False,
                "seam_gap": 0.125,
                "min_leg": 6.0,
                "prefer_long_split": True,
                "enable_L_seams": True,
                "L_max_leg": 48.0,
                "grid_step": 0.5,
                "units": "in",
                "precision": 2
            }
        }


class NestingRequest(BaseModel):
    """Request body for nesting operations"""
    parts: List[Part]
    config: NestingConfig
    mode: str = Field(default="rectpack", description="'rectpack' or 'polygon'")


class PlacementInfo(BaseModel):
    """Information about a placed part"""
    x: float
    y: float
    w: float
    h: float
    rid: str
    label: str


class SheetInfo(BaseModel):
    """Information about a sheet with placements"""
    sheet_w: float
    sheet_h: float
    placements: List[PlacementInfo]


class NestingResult(BaseModel):
    """Response for nesting operations"""
    success: bool = True
    sheets: List[Dict[str, Any]]
    utilization: float
    num_sheets: int
    total_area_used: float
    total_area_available: float
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response format"""
    success: bool = False
    error: str
    detail: Optional[str] = None
