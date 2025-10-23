# app.py — Enhanced Nesting Tool with all improvements
from __future__ import annotations

import math
import uuid
import json
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
import io

import pandas as pd
import plotly.graph_objects as go
from rectpack import newPacker
from shapely.affinity import rotate as shp_rotate, translate as shp_translate
from shapely.geometry import Polygon, box as shp_box

import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ============================================================================
# PAGE SETUP
# ============================================================================
st.set_page_config(page_title="Nesting Tool Pro", layout="wide", page_icon="🔲")

# ============================================================================
# DATA MODELS
# ============================================================================
@dataclass
class Part:
    """Represents a part to be nested"""
    id: str
    label: str
    qty: int
    shape_type: str  # "rect" or "polygon"
    width: Optional[float]
    height: Optional[float]
    points: Optional[List[Tuple[float, float]]]
    allow_rotation: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def to_polygon(self) -> Optional[Polygon]:
        """Convert part to Shapely polygon"""
        try:
            if self.shape_type == "rect" and self.width and self.height:
                poly = shp_box(0, 0, self.width, self.height)
                if self.meta.get("cutouts"):
                    for (cx, cy, cw, ch) in self.meta["cutouts"]:
                        poly = poly.difference(shp_box(cx, cy, cx + cw, cy + ch))
                return poly
            elif self.shape_type == "polygon" and self.points:
                poly = Polygon(self.points)
                if self.meta.get("cutouts"):
                    for (cx, cy, cw, ch) in self.meta["cutouts"]:
                        poly = poly.difference(shp_box(cx, cy, cx + cw, cy + ch))
                return poly
        except Exception as e:
            st.warning(f"Error converting {self.label} to polygon: {e}")
        return None
    
    def get_dimensions(self) -> Tuple[float, float]:
        """Safely get part dimensions"""
        try:
            if self.shape_type == "rect":
                return self.width or 0.0, self.height or 0.0
            elif self.points:
                poly = Polygon(self.points)
                bounds = poly.bounds
                return bounds[2] - bounds[0], bounds[3] - bounds[1]
        except Exception:
            return 0.0, 0.0
        return 0.0, 0.0


@dataclass
class NestingConfig:
    """Configuration for nesting operations"""
    sheet_w: float
    sheet_h: float
    clearance: float
    allow_rotation: bool
    autosplit_rects: bool
    seam_gap: float
    min_leg: float
    prefer_long_split: bool
    enable_L_seams: bool
    L_max_leg: float
    grid_step: float = 0.5
    units: str = "in"
    precision: int = 2
    
    @classmethod
    def from_session_state(cls) -> 'NestingConfig':
        """Create config from Streamlit session state"""
        return cls(
            sheet_w=st.session_state.get("sb_sheet_w", 97.0),
            sheet_h=st.session_state.get("sb_sheet_h", 80.5),
            clearance=st.session_state.get("sb_clearance", 0.25),
            allow_rotation=st.session_state.get("sb_allow_rot", True),
            autosplit_rects=st.session_state.get("sb_autosplit", True),
            seam_gap=st.session_state.get("sb_seam_gap", 0.125),
            min_leg=st.session_state.get("sb_min_leg", 6.0),
            prefer_long_split=st.session_state.get("sb_prefer_split", "long side") == "long side",
            enable_L_seams=st.session_state.get("sb_L_seams", True),
            L_max_leg=st.session_state.get("sb_L_max_leg", 48.0),
            units=st.session_state.get("sb_units", "in"),
            precision=st.session_state.get("sb_precision", 2),
        )


class NestingError(Exception):
    """Custom exception for nesting failures"""
    pass


# ============================================================================
# STATE MANAGEMENT
# ============================================================================
def _init_state():
    """Initialize session state with all required keys"""
    defaults = {
        "parts": [],
        "needs_nest": True,
        "placements": [],
        "utilization": 0.0,
        "messages": [],
        "draw_canvas_key": "draw_canvas",
        "history": [],
        "history_index": -1,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def save_to_history():
    """Save current state to history for undo/redo"""
    state_snapshot = {
        "parts": [asdict(p) for p in st.session_state.parts],
        "placements": st.session_state.placements.copy(),
        "utilization": st.session_state.utilization,
    }
    
    # Trim future history if we're not at the end
    if st.session_state.history_index < len(st.session_state.history) - 1:
        st.session_state.history = st.session_state.history[:st.session_state.history_index + 1]
    
    st.session_state.history.append(state_snapshot)
    st.session_state.history_index += 1
    
    # Keep only last 50 states to avoid memory issues
    if len(st.session_state.history) > 50:
        st.session_state.history.pop(0)
        st.session_state.history_index -= 1


def restore_from_history(index: int):
    """Restore state from history"""
    if 0 <= index < len(st.session_state.history):
        snapshot = st.session_state.history[index]
        st.session_state.parts = [Part(**p) for p in snapshot["parts"]]
        st.session_state.placements = snapshot["placements"]
        st.session_state.utilization = snapshot["utilization"]
        st.session_state.history_index = index
        st.session_state.needs_nest = True


_init_state()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def _pretty_units(u: str) -> str:
    """Format units for display"""
    return {"in": "in", "mm": "mm", "cm": "cm"}.get(u, u)


def validate_part(part: Part, sheet_w: float, sheet_h: float) -> Tuple[bool, str]:
    """Validate part before adding"""
    if part.shape_type == "rect":
        if not part.width or not part.height or part.width <= 0 or part.height <= 0:
            return False, "Width and height must be positive"
        if part.width > sheet_w * 10 or part.height > sheet_h * 10:
            return False, "Part dimensions unreasonably large (>10x sheet size)"
    
    if part.qty <= 0:
        return False, "Quantity must be positive"
    
    if part.shape_type == "polygon" and part.points:
        if len(part.points) < 3:
            return False, "Polygon must have at least 3 points"
    
    return True, "OK"


@st.cache_data
def compute_polygon_from_points(points: tuple, cutouts: tuple = None) -> Polygon:
    """Cache polygon creation - points must be immutable (tuple)"""
    poly = Polygon(points)
    if cutouts:
        for (cx, cy, cw, ch) in cutouts:
            poly = poly.difference(shp_box(cx, cy, cx + cw, cy + ch))
    return poly


# ============================================================================
# CANVAS HELPERS
# ============================================================================
def _fabric_rect_dims(obj: Dict) -> Tuple[float, float]:
    """Extract rectangle dimensions from Fabric.js object"""
    w = float(obj.get("width", 0)) * float(obj.get("scaleX", 1.0))
    h = float(obj.get("height", 0)) * float(obj.get("scaleY", 1.0))
    return abs(w), abs(h)


def _fabric_polygon_points(obj: Dict) -> List[Tuple[float, float]]:
    """Extract polygon points from Fabric.js object"""
    pts = []
    path = obj.get("path")
    if isinstance(path, list):
        x0 = float(obj.get("left", 0))
        y0 = float(obj.get("top", 0))
        sx = float(obj.get("scaleX", 1.0))
        sy = float(obj.get("scaleY", 1.0))
        for seg in path:
            if isinstance(seg, list) and len(seg) >= 3 and seg[0] in ("L", "M"):
                px = x0 + float(seg[1]) * sx
                py = y0 + float(seg[2]) * sy
                pts.append((px, py))
    return pts


def _bbox_of_polygon(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Calculate bounding box of polygon"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (max(xs) - min(xs), max(ys) - min(ys))


def polygon_with_cutouts(
    outer_pts: List[Tuple[float, float]], 
    cutouts: List[Tuple[float, float, float, float]]
) -> Polygon:
    """Create polygon with rectangular cutouts"""
    poly = Polygon(outer_pts)
    for (cx, cy, cw, ch) in cutouts:
        poly = poly.difference(shp_box(cx, cy, cx + cw, cy + ch))
    return poly


# ============================================================================
# PROJECT IMPORT/EXPORT
# ============================================================================
def export_project(config: NestingConfig) -> str:
    """Export entire project as JSON"""
    project = {
        "version": "1.0",
        "created": datetime.now().isoformat(),
        "config": asdict(config),
        "parts": [asdict(p) for p in st.session_state.parts],
        "placements": st.session_state.placements,
        "utilization": st.session_state.utilization,
    }
    return json.dumps(project, indent=2)


def import_project(json_data: str) -> bool:
    """Import project from JSON"""
    try:
        project = json.loads(json_data)
        
        # Restore parts
        st.session_state.parts = [Part(**p) for p in project.get("parts", [])]
        st.session_state.placements = project.get("placements", [])
        st.session_state.utilization = project.get("utilization", 0.0)
        st.session_state.needs_nest = False
        
        # Restore config to session state
        config = project.get("config", {})
        for key, value in config.items():
            if key in ["sheet_w", "sheet_h", "clearance", "seam_gap", "min_leg", "L_max_leg"]:
                st.session_state[f"sb_{key}"] = value
        
        save_to_history()
        return True
    except Exception as e:
        st.error(f"Failed to import project: {e}")
        return False


def export_cutting_list() -> pd.DataFrame:
    """Generate cutting list DataFrame"""
    rows = []
    for p in st.session_state.parts:
        w, h = p.get_dimensions()
        rows.append({
            "Label": p.label,
            "Type": "L" if p.meta.get("is_L") else p.shape_type,
            "Width": round(w, 2),
            "Height": round(h, 2),
            "Quantity": p.qty,
            "Total Area": round(w * h * p.qty, 2),
            "Rotation Allowed": "Yes" if p.allow_rotation else "No",
        })
    return pd.DataFrame(rows)


# ============================================================================
# SPLITTING & DECOMPOSITION
# ============================================================================
def split_rect_if_needed(
    p: Part, 
    config: NestingConfig
) -> Tuple[List[Part], bool]:
    """Split rectangle if it exceeds sheet dimensions"""
    if p.shape_type != "rect" or p.width is None or p.height is None:
        return [p], False
    
    W, H = p.width, p.height
    usable_w = max(0.0, config.sheet_w - config.clearance)
    usable_h = max(0.0, config.sheet_h - config.clearance)
    
    if W <= usable_w and H <= usable_h:
        return [p], False
    
    parts_out: List[Part] = []
    long_side = "W" if W >= H else "H"
    split_along_width = (long_side == "W") if config.prefer_long_split else (long_side == "H")
    
    if W > usable_w and H <= usable_h:
        split_along_width = True
    if H > usable_h and W <= usable_w:
        split_along_width = False
    
    if split_along_width:
        remaining = W
        idx = 1
        while remaining > 1e-9:
            panel = min(usable_w, remaining)
            if panel < config.min_leg - 1e-9:
                if parts_out:
                    prev = parts_out.pop()
                    merged = prev.width + config.seam_gap + panel
                    if merged > usable_w + 1e-9:
                        st.session_state.messages.append(
                            f"⚠️ Cannot split {p.label}: panel too short."
                        )
                        return [p], False
                    parts_out.append(Part(
                        id=str(uuid.uuid4()),
                        label=f"{p.label}-S{idx-1}",
                        qty=p.qty,
                        shape_type="rect",
                        width=merged,
                        height=H,
                        points=None,
                        allow_rotation=p.allow_rotation,
                        meta={"from": p.id}
                    ))
                else:
                    st.session_state.messages.append(
                        f"⚠️ Cannot split {p.label}: min leg violated."
                    )
                    return [p], False
            else:
                parts_out.append(Part(
                    id=str(uuid.uuid4()),
                    label=f"{p.label}-S{idx}",
                    qty=p.qty,
                    shape_type="rect",
                    width=panel,
                    height=H,
                    points=None,
                    allow_rotation=p.allow_rotation,
                    meta={"from": p.id}
                ))
            remaining -= panel
            if remaining > 1e-9:
                remaining -= config.seam_gap
            idx += 1
    else:
        remaining = H
        idx = 1
        while remaining > 1e-9:
            panel = min(usable_h, remaining)
            if panel < config.min_leg - 1e-9:
                if parts_out:
                    prev = parts_out.pop()
                    merged = prev.height + config.seam_gap + panel
                    if merged > usable_h + 1e-9:
                        st.session_state.messages.append(
                            f"⚠️ Cannot split {p.label}: panel too short."
                        )
                        return [p], False
                    parts_out.append(Part(
                        id=str(uuid.uuid4()),
                        label=f"{p.label}-S{idx-1}",
                        qty=p.qty,
                        shape_type="rect",
                        width=W,
                        height=merged,
                        points=None,
                        allow_rotation=p.allow_rotation,
                        meta={"from": p.id}
                    ))
                else:
                    st.session_state.messages.append(
                        f"⚠️ Cannot split {p.label}: min leg violated."
                    )
                    return [p], False
            else:
                parts_out.append(Part(
                    id=str(uuid.uuid4()),
                    label=f"{p.label}-S{idx}",
                    qty=p.qty,
                    shape_type="rect",
                    width=W,
                    height=panel,
                    points=None,
                    allow_rotation=p.allow_rotation,
                    meta={"from": p.id}
                ))
            remaining -= panel
            if remaining > 1e-9:
                remaining -= config.seam_gap
            idx += 1
    
    return (parts_out if parts_out else [p]), (len(parts_out) > 1)


def decompose_L_prefer_corner(p: Part) -> List[Part]:
    """Decompose L-shape by removing corner square"""
    m = p.meta or {}
    if not m.get("is_L"):
        return [p]
    if abs(float(m.get("angle", 90.0)) - 90.0) > 1e-6:
        return [p]
    
    A = float(m.get("A", 0.0))
    B = float(m.get("B", 0.0))
    D = float(m.get("tA", 0.0))
    
    r1 = Part(
        id=str(uuid.uuid4()),
        label=f"{p.label}-legA",
        qty=p.qty,
        shape_type="rect",
        width=max(0.0, A - D),
        height=D,
        points=None,
        allow_rotation=p.allow_rotation,
        meta={"from_L": p.id, "leg": "A"}
    )
    r2 = Part(
        id=str(uuid.uuid4()),
        label=f"{p.label}-legB",
        qty=p.qty,
        shape_type="rect",
        width=D,
        height=B,
        points=None,
        allow_rotation=p.allow_rotation,
        meta={"from_L": p.id, "leg": "B"}
    )
    
    if r1.width <= 0 or r1.height <= 0 or r2.width <= 0 or r2.height <= 0:
        return [p]
    return [r1, r2]


def decompose_L_alternate(p: Part) -> List[Part]:
    """Decompose L-shape alternative method"""
    m = p.meta or {}
    if not m.get("is_L"):
        return [p]
    if abs(float(m.get("angle", 90.0)) - 90.0) > 1e-6:
        return [p]
    
    A = float(m.get("A", 0.0))
    B = float(m.get("B", 0.0))
    D = float(m.get("tA", 0.0))
    
    r1 = Part(
        id=str(uuid.uuid4()),
        label=f"{p.label}-legA",
        qty=p.qty,
        shape_type="rect",
        width=A,
        height=D,
        points=None,
        allow_rotation=p.allow_rotation,
        meta={"from_L": p.id, "leg": "A"}
    )
    r2 = Part(
        id=str(uuid.uuid4()),
        label=f"{p.label}-legB",
        qty=p.qty,
        shape_type="rect",
        width=max(0.0, B - D),
        height=D,
        points=None,
        allow_rotation=p.allow_rotation,
        meta={"from_L": p.id, "leg": "B"}
    )
    
    if r1.width <= 0 or r1.height <= 0 or r2.width <= 0 or r2.height <= 0:
        return [p]
    return [r1, r2]


# ============================================================================
# NESTING ENGINES
# ============================================================================
def _rectpack_trial(parts_trial, sheet_w, sheet_h, clearance, rotation):
    """Trial run of rectpack for comparison"""
    packer = newPacker(rotation=rotation)
    EPS = 1e-6
    for s in parts_trial:
        w = min((s.width or 0) + clearance, sheet_w - EPS)
        h = min((s.height or 0) + clearance, sheet_h - EPS)
        packer.add_rect(w, h, rid=str(uuid.uuid4()))
    for _ in range(50):
        packer.add_bin(sheet_w, sheet_h)
    packer.pack()
    return [abin for abin in packer if abin.rect_list()], None


def rectpack_nest(parts: List[Part], config: NestingConfig):
    """Fast rectangular nesting using rectpack"""
    expanded: List[Part] = []
    
    for p in parts:
        if p.qty <= 0:
            continue
        
        subs = [p]
        did_split = False
        
        # Handle L-shapes
        if p.meta.get("is_L") and abs(p.meta.get("angle", 90.0) - 90.0) < 1e-6:
            cand1 = decompose_L_prefer_corner(p)
            cand2 = decompose_L_alternate(p)
            s1, _ = _rectpack_trial(cand1, config.sheet_w, config.sheet_h, config.clearance, config.allow_rotation)
            s2, _ = _rectpack_trial(cand2, config.sheet_w, config.sheet_h, config.clearance, config.allow_rotation)
            subs = cand1 if len(s1) <= len(s2) else cand2
            did_split = True
        
        # Handle autosplit
        if config.autosplit_rects:
            next_subs = []
            for s in subs:
                if s.shape_type == "rect":
                    ss, did = split_rect_if_needed(s, config)
                    did_split = did_split or did
                    next_subs.extend(ss)
                else:
                    next_subs.append(s)
            subs = next_subs
        
        for s in subs:
            for _ in range(s.qty):
                expanded.append(s)
    
    packer = newPacker(rotation=config.allow_rotation)
    EPS = 1e-6
    to_add = []
    
    for s in expanded:
        if s.shape_type == "rect" and s.width and s.height:
            w = min(s.width + config.clearance, config.sheet_w - EPS)
            h = min(s.height + config.clearance, config.sheet_h - EPS)
        else:
            if s.points:
                poly = Polygon(s.points)
                minx, miny, maxx, maxy = poly.bounds
                w = min((maxx - minx) + config.clearance, config.sheet_w - EPS)
                h = min((maxy - miny) + config.clearance, config.sheet_h - EPS)
            else:
                continue
        
        rid = f"{s.label}#{uuid.uuid4().hex[:4]}"
        to_add.append((w, h, rid, s.label))
    
    if not to_add:
        return [], 0.0
    
    for (w, h, rid, _) in to_add:
        packer.add_rect(w, h, rid=rid)
    
    for _ in range(200):
        packer.add_bin(config.sheet_w, config.sheet_h)
    
    packer.pack()
    
    sheets = []
    total_area = 0.0
    
    for abin in packer:
        rects = abin.rect_list()
        if not rects:
            continue
        placements = []
        for (x, y, w, h, rid) in rects:
            label = rid.split("#")[0]
            placements.append({
                "x": x,
                "y": y,
                "w": w - config.clearance,
                "h": h - config.clearance,
                "rid": rid,
                "label": label
            })
            total_area += max(0.0, (w - config.clearance)) * max(0.0, (h - config.clearance))
        sheets.append({
            "sheet_w": config.sheet_w,
            "sheet_h": config.sheet_h,
            "placements": placements
        })
    
    util = total_area / max(1e-9, len(sheets) * config.sheet_w * config.sheet_h)
    return sheets, util


def _poly_to_rect_anno(poly: Polygon, label: str, precision: int):
    """Convert polygon to rectangular annotation"""
    minx, miny, maxx, maxy = poly.bounds
    w = maxx - minx
    h = maxy - miny
    return {
        "x": float(minx),
        "y": float(miny),
        "w": float(w),
        "h": float(h),
        "rid": f"{label}#{uuid.uuid4().hex[:4]}",
        "label": label,
        "exact_w": round(w, precision),
        "exact_h": round(h, precision)
    }


def poly_nest(parts: List[Part], config: NestingConfig):
    """Precision polygon-aware nesting"""
    expanded: List[Tuple[Part, Polygon]] = []
    
    for p in parts:
        if p.qty <= 0:
            continue
        
        subs = [p]
        did_split = False
        
        # Handle L-shapes
        if p.meta.get("is_L") and abs(p.meta.get("angle", 90.0) - 90.0) < 1e-6:
            cand1 = decompose_L_prefer_corner(p)
            cand2 = decompose_L_alternate(p)
            s1, _ = _rectpack_trial(cand1, config.sheet_w, config.sheet_h, config.clearance, config.allow_rotation)
            s2, _ = _rectpack_trial(cand2, config.sheet_w, config.sheet_h, config.clearance, config.allow_rotation)
            subs = cand1 if len(s1) <= len(s2) else cand2
            did_split = True
        
        # Handle autosplit
        if config.autosplit_rects:
            next_subs = []
            for s in subs:
                if s.shape_type == "rect":
                    ss, did = split_rect_if_needed(s, config)
                    did_split = did_split or did
                    next_subs.extend(ss)
                else:
                    next_subs.append(s)
            subs = next_subs
        
        for s in subs:
            shp = s.to_polygon()
            if shp is None:
                continue
            for _ in range(s.qty):
                expanded.append((s, shp))
    
    # Sort by area (largest first) for better packing
    expanded.sort(key=lambda t: t[1].area, reverse=True)
    
    placed = []
    sheet_poly = shp_box(0, 0, config.sheet_w, config.sheet_h)
    buffer_clear = config.clearance / 2.0 if config.clearance > 0 else 0.0
    
    # Adaptive grid step based on smallest dimension
    min_dim = min(s[1].bounds[2] - s[1].bounds[0] for s in expanded) if expanded else 1.0
    grid_step = max(0.25, min_dim / 10)
    
    for s, base_poly in expanded:
        orientations = [0] + ([90] if config.allow_rotation else [])
        placed_ok = False
        
        for ang in orientations:
            poly = shp_rotate(base_poly, ang, origin=(0, 0), use_radians=False)
            bounds = poly.bounds
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            
            # Quick reject if too large
            if width > config.sheet_w or height > config.sheet_h:
                continue
            
            y = 0.0
            while y <= config.sheet_h - height and not placed_ok:
                x = 0.0
                while x <= config.sheet_w - width and not placed_ok:
                    test = shp_translate(poly, xoff=x, yoff=y)
                    
                    # Check if within sheet
                    if not sheet_poly.covers(test):
                        x += grid_step
                        continue
                    
                    # Check collisions
                    collision = any(
                        test.buffer(buffer_clear).intersects(o["poly"].buffer(buffer_clear))
                        for o in placed
                    )
                    
                    if not collision:
                        placed.append({
                            "poly": test,
                            "label": s.label,
                            "rid": f"{s.label}#{uuid.uuid4().hex[:4]}",
                            "angle": ang
                        })
                        placed_ok = True
                        break
                    
                    x += grid_step
                y += grid_step
            
            if placed_ok:
                break
        
        if not placed_ok:
            # Start new sheet
            yield {
                "sheet_w": config.sheet_w,
                "sheet_h": config.sheet_h,
                "placements": [
                    _poly_to_rect_anno(d["poly"], d["label"], config.precision)
                    for d in placed
                ]
            }
            placed = []
            
            # Try to place on fresh sheet
            for ang in ([0, 90] if config.allow_rotation else [0]):
                ptry = shp_rotate(base_poly, ang, origin=(0, 0), use_radians=False)
                if ptry.within(sheet_poly):
                    placed.append({
                        "poly": ptry,
                        "label": s.label,
                        "rid": f"{s.label}#{uuid.uuid4().hex[:4]}",
                        "angle": ang
                    })
                    break
            
            if not placed:
                st.session_state.messages.append(
                    f"⚠️ {s.label}: cannot place on empty sheet (too large)."
                )
    
    if placed:
        yield {
            "sheet_w": config.sheet_w,
            "sheet_h": config.sheet_h,
            "placements": [
                _poly_to_rect_anno(d["poly"], d["label"], config.precision)
                for d in placed
            ]
        }


# ============================================================================
# L-SHAPE GENERATOR
# ============================================================================
def make_L_polygon(A: float, D: float, B: float, angle_deg: float) -> List[Tuple[float, float]]:
    """Generate L-shape polygon points"""
    pts = [(0, 0), (A, 0), (A, D), (D, D), (D, B), (0, B), (0, 0)]
    if abs(angle_deg - 90.0) > 1e-6:
        theta = math.radians(90.0 - angle_deg)
        rot = lambda p: (
            p[0] * math.cos(theta) - p[1] * math.sin(theta),
            p[0] * math.sin(theta) + p[1] * math.cos(theta)
        )
        pts = [rot(p) for p in pts]
    return pts


# ============================================================================
# UI COMPONENTS
# ============================================================================
def render_sidebar(config: NestingConfig):
    """Render sidebar with project settings"""
    with st.sidebar:
        st.header("⚙️ Project Settings")
        
        # Units and precision
        units = st.selectbox("Units", ["in", "mm", "cm"], index=0, key="sb_units")
        precision = st.number_input(
            "Precision (decimals)", 0, 4, 2, key="sb_precision"
        )
        px_per_unit = st.slider(
            "Canvas scale (px per unit)", 2, 20, 8, key="sb_ppu"
        )
        
        st.markdown("### 📐 Drawing Canvas Size")
        canvas_w = st.number_input(
            "Canvas width (px)", 
            min_value=600, 
            max_value=2400, 
            value=1400, 
            step=50, 
            key="sb_canvas_w"
        )
        canvas_h = st.number_input(
            "Canvas height (px)", 
            min_value=400, 
            max_value=2000, 
            value=950, 
            step=50, 
            key="sb_canvas_h"
        )
        
        st.markdown("### 📄 Sheet Settings")
        sheet_w = st.number_input(
            f"Sheet width ({_pretty_units(units)})",
            min_value=1.0,
            value=97.0,
            step=1.0,
            format="%.2f",
            key="sb_sheet_w"
        )
        sheet_h = st.number_input(
            f"Sheet height ({_pretty_units(units)})",
            min_value=1.0,
            value=80.50,
            step=1.0,
            format="%.2f",
            key="sb_sheet_h"
        )
        clearance = st.number_input(
            f"Clearance between parts ({_pretty_units(units)})",
            min_value=0.0,
            value=0.25,
            step=0.05,
            format="%.2f",
            key="sb_clearance"
        )
        allow_rotation_global = st.checkbox(
            "Allow rotation globally (0/90°)", value=True, key="sb_allow_rot"
        )
        
        st.markdown("### 🔄 Nesting Mode")
        mode = st.radio(
            "Mode",
            ["Fast (rectpack)", "Precision (polygon-aware)"],
            index=1,
            key="sb_mode"
        )
        
        st.markdown("### ✂️ Auto-Split Settings")
        autosplit_rects = st.checkbox(
            "Auto-split rectangles that exceed sheet size",
            value=True,
            key="sb_autosplit"
        )
        seam_gap = st.number_input(
            f"Seam/kerf gap at split ({_pretty_units(units)})",
            min_value=0.0,
            value=0.125,
            step=0.01,
            format="%.3f",
            key="sb_seam_gap"
        )
        min_leg = st.number_input(
            f"Minimum leg length after split ({_pretty_units(units)})",
            min_value=1.0,
            value=6.0,
            step=0.5,
            format="%.2f",
            key="sb_min_leg"
        )
        prefer_long_split = st.selectbox(
            "Prefer split along",
            ["long side", "short side"],
            index=0,
            key="sb_prefer_split"
        )
        
        st.markdown("### 🔲 L-Shape Split Policy")
        enable_L_seams = st.checkbox(
            "Allow seams on L-shapes when needed",
            value=True,
            key="sb_L_seams"
        )
        L_max_leg_no_split = st.number_input(
            f"Max leg length without split ({_pretty_units(units)})",
            min_value=1.0,
            value=48.0,
            step=1.0,
            format="%.0f",
            key="sb_L_max_leg"
        )
        
        st.markdown("---")
        
        # Project management
        st.markdown("### 💾 Project Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Export", use_container_width=True):
                config = NestingConfig.from_session_state()
                json_str = export_project(config)
                st.download_button(
                    "⬇️ Download JSON",
                    data=json_str,
                    file_name=f"nesting_{datetime.now():%Y%m%d_%H%M%S}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col2:
            uploaded = st.file_uploader(
                "📁 Import",
                type=['json'],
                key="import_file",
                label_visibility="collapsed"
            )
            if uploaded:
                json_data = uploaded.read().decode('utf-8')
                if import_project(json_data):
                    st.success("✅ Project imported!")
                    st.rerun()
        
        # Export cutting list
        if st.session_state.parts:
            cutting_list = export_cutting_list()
            csv = cutting_list.to_csv(index=False)
            st.download_button(
                "📋 Export Cutting List (CSV)",
                data=csv,
                file_name=f"cutting_list_{datetime.now():%Y%m%d_%H%M%S}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        st.markdown("---")
        
        # History controls
        st.markdown("### ↶ History")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "↶ Undo",
                disabled=st.session_state.history_index <= 0,
                use_container_width=True
            ):
                restore_from_history(st.session_state.history_index - 1)
                st.rerun()
        with col2:
            if st.button(
                "↷ Redo",
                disabled=st.session_state.history_index >= len(st.session_state.history) - 1,
                use_container_width=True
            ):
                restore_from_history(st.session_state.history_index + 1)
                st.rerun()
        
        st.caption(f"Step {st.session_state.history_index + 1} of {len(st.session_state.history)}")
        
        st.markdown("---")
        
        # Clear project
        if st.button("🗑️ Clear Project", type="primary", use_container_width=True):
            for k in ["parts", "needs_nest", "placements", "utilization", "messages", "draw_canvas_key", "history", "history_index"]:
                st.session_state.pop(k, None)
            _init_state()
            st.success("✅ Project cleared.")
            st.rerun()
        
        # Debug mode
        if st.checkbox("🐛 Debug Mode"):
            with st.expander("Debug Info"):
                st.write("Parts count:", len(st.session_state.parts))
                st.write("History steps:", len(st.session_state.history))
                st.write("Needs nesting:", st.session_state.needs_nest)


def render_canvas_and_tools(px_per_unit: int, canvas_w: int, canvas_h: int, precision: int, units: str):
    """Render drawing canvas and tool controls"""
    tool = st.radio(
        "🔧 Tool",
        ["Rectangle", "Polygon (freehand)", "L-shape (parametric)"],
        horizontal=True,
        key="tool_radio"
    )
    
    # Drawing canvas
    drawing_mode = "rect" if tool == "Rectangle" else "polygon"
    canvas_key = st.session_state.get("draw_canvas_key", "draw_canvas")
    
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0.0)",
        stroke_width=2,
        stroke_color="#1f77b4",
        background_color="#fafafa",
        height=int(canvas_h),
        width=int(canvas_w),
        drawing_mode=drawing_mode,
        display_toolbar=True,
        update_streamlit=True,
        key=canvas_key
    )
    
    # Live measurements
    last_obj = None
    live_w = live_h = None
    
    if canvas_result and canvas_result.json_data:
        objs = canvas_result.json_data.get("objects", [])
        if objs:
            last_obj = objs[-1]
            if drawing_mode == "rect" and last_obj.get("type") == "rect":
                w_px, h_px = _fabric_rect_dims(last_obj)
                live_w = round(abs(w_px) / px_per_unit, precision)
                live_h = round(abs(h_px) / px_per_unit, precision)
            elif drawing_mode != "rect" and last_obj.get("path"):
                pts = _fabric_polygon_points(last_obj)
                if len(pts) >= 3:
                    bw, bh = _bbox_of_polygon(pts)
                    live_w = round(bw / px_per_unit, precision)
                    live_h = round(bh / px_per_unit, precision)
    
    col_live = st.columns(2)
    with col_live[0]:
        st.metric(
            "Live Width",
            f"{live_w if live_w is not None else '—'} {_pretty_units(units)}"
        )
    with col_live[1]:
        st.metric(
            "Live Height",
            f"{live_h if live_h is not None else '—'} {_pretty_units(units)}"
        )
    
    return tool, canvas_result, last_obj, px_per_unit


def render_add_shape_controls(tool: str, last_obj: Any, canvas_result: Any, px_per_unit: int, precision: int, units: str):
    """Render controls for adding shapes"""
    config = NestingConfig.from_session_state()
    
    with st.expander("➕ Add Shape to Parts List", expanded=True):
        qty = st.number_input("Quantity", min_value=1, step=1, value=1, key="exp_qty")
        label = st.text_input("Label (optional)", value="", key="exp_label")
        allow_rot = st.checkbox(
            "Allow rotation for this part (0/90°)",
            value=True,
            key="exp_rot"
        )
        
        st.markdown("**Optional: Rectangular Cutouts**")
        add_cut = st.checkbox("Add cutout(s) to this part", key="exp_cut_toggle")
        cut_list: List[Tuple[float, float, float, float]] = []
        
        if add_cut:
            ncuts = st.number_input(
                "Number of cutouts",
                min_value=1,
                max_value=5,
                value=1,
                step=1,
                key="exp_cut_count"
            )
            for i in range(ncuts):
                st.write(f"**Cutout #{i+1}**")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    cw = st.number_input(
                        f"W{i+1}",
                        min_value=0.1,
                        value=30.0,
                        step=0.1,
                        format="%.2f",
                        key=f"exp_cw_{i}"
                    )
                with c2:
                    ch = st.number_input(
                        f"H{i+1}",
                        min_value=0.1,
                        value=20.0,
                        step=0.1,
                        format="%.2f",
                        key=f"exp_ch_{i}"
                    )
                with c3:
                    cx = st.number_input(
                        f"X{i+1}",
                        value=10.0,
                        step=0.5,
                        format="%.2f",
                        key=f"exp_cx_{i}"
                    )
                with c4:
                    cy = st.number_input(
                        f"Y{i+1}",
                        value=10.0,
                        step=0.5,
                        format="%.2f",
                        key=f"exp_cy_{i}"
                    )
                cut_list.append((cx, cy, cw, ch))
        
        if st.button("➕ Add This Shape", type="secondary", key="exp_add_btn"):
            new_part = None
            
            if tool == "Rectangle":
                if not last_obj or last_obj.get("type") != "rect":
                    st.warning("⚠️ Draw a rectangle first.")
                else:
                    w_px, h_px = _fabric_rect_dims(last_obj)
                    w = round(w_px / px_per_unit, precision)
                    h = round(h_px / px_per_unit, precision)
                    
                    if w <= 0 or h <= 0:
                        st.error("❌ Rectangle has zero width/height.")
                    else:
                        if add_cut and cut_list:
                            outer = [(0, 0), (w, 0), (w, h), (0, h), (0, 0)]
                            poly = polygon_with_cutouts(outer, cut_list)
                            new_part = Part(
                                id=str(uuid.uuid4()),
                                label=label or f"Rect-{len(st.session_state.parts)+1}",
                                qty=int(qty),
                                shape_type="polygon",
                                width=None,
                                height=None,
                                points=list(poly.exterior.coords),
                                allow_rotation=bool(allow_rot),
                                meta={"cutouts": cut_list}
                            )
                        else:
                            new_part = Part(
                                id=str(uuid.uuid4()),
                                label=label or f"Rect-{len(st.session_state.parts)+1}",
                                qty=int(qty),
                                shape_type="rect",
                                width=float(w),
                                height=float(h),
                                points=None,
                                allow_rotation=bool(allow_rot)
                            )
            
            elif tool == "Polygon (freehand)":
                if not last_obj or not last_obj.get("path"):
                    st.warning("⚠️ Draw a polygon first.")
                else:
                    pts = _fabric_polygon_points(last_obj)
                    if len(pts) < 3:
                        st.error("❌ Polygon needs at least 3 points.")
                    else:
                        pts_units = [(x / px_per_unit, y / px_per_unit) for (x, y) in pts]
                        new_part = Part(
                            id=str(uuid.uuid4()),
                            label=label or f"Poly-{len(st.session_state.parts)+1}",
                            qty=int(qty),
                            shape_type="polygon",
                            width=None,
                            height=None,
                            points=pts_units,
                            allow_rotation=bool(allow_rot)
                        )
            
            if new_part:
                is_valid, msg = validate_part(new_part, config.sheet_w, config.sheet_h)
                if is_valid:
                    st.session_state.parts = st.session_state.parts + [new_part]
                    st.session_state.needs_nest = True
                    save_to_history()
                    st.success(f"✅ Added {new_part.label}")
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")
    
    # Bulk operations
    bulk_cols = st.columns([1, 1, 1])
    with bulk_cols[0]:
        if st.button("➕ Add All Drawn Shapes", type="secondary", key="btn_add_all_drawn"):
            added = 0
            if canvas_result and canvas_result.json_data:
                objs = canvas_result.json_data.get("objects", [])
                new_parts = []
                
                for obj in objs:
                    if obj.get("type") == "rect":
                        w_px, h_px = _fabric_rect_dims(obj)
                        w = round(w_px / px_per_unit, precision)
                        h = round(h_px / px_per_unit, precision)
                        if w > 0 and h > 0:
                            new_parts.append(Part(
                                id=str(uuid.uuid4()),
                                label=f"Draw-{len(st.session_state.parts)+added+1}",
                                qty=1,
                                shape_type="rect",
                                width=float(w),
                                height=float(h),
                                points=None,
                                allow_rotation=True
                            ))
                            added += 1
                    elif obj.get("path"):
                        pts = _fabric_polygon_points(obj)
                        if len(pts) >= 3:
                            pts_units = [(x / px_per_unit, y / px_per_unit) for (x, y) in pts]
                            new_parts.append(Part(
                                id=str(uuid.uuid4()),
                                label=f"Draw-{len(st.session_state.parts)+added+1}",
                                qty=1,
                                shape_type="polygon",
                                width=None,
                                height=None,
                                points=pts_units,
                                allow_rotation=True
                            ))
                            added += 1
                
                if new_parts:
                    st.session_state.parts = st.session_state.parts + new_parts
                    st.session_state.needs_nest = True
                    save_to_history()
                    st.success(f"✅ Added {added} shape(s)")
                    st.rerun()
    
    with bulk_cols[1]:
        if st.button("🧹 Clear Drawing Layer", key="btn_clear_layer"):
            st.session_state["draw_canvas_key"] = uuid.uuid4().hex
            st.rerun()
    
    with bulk_cols[2]:
        st.caption("💡 Tip: Draw multiple shapes, then **Add All**")


def render_L_shape_tool(precision: int, units: str):
    """Render L-shape parametric tool"""
    st.markdown("### 🔲 Create L-Shape")
    
    colA, colB = st.columns(2)
    with colA:
        A = st.number_input(
            f"Outer leg A length ({_pretty_units(units)})",
            min_value=1.0,
            value=120.0,
            step=0.5,
            format="%.2f",
            key="L_A"
        )
        D = st.number_input(
            f"Depth / thickness D ({_pretty_units(units)})",
            min_value=0.1,
            value=25.0,
            step=0.1,
            format="%.2f",
            key="L_D"
        )
    with colB:
        B = st.number_input(
            f"Outer leg B length ({_pretty_units(units)})",
            min_value=1.0,
            value=60.0,
            step=0.5,
            format="%.2f",
            key="L_B"
        )
    
    ang = st.number_input(
        "Inside angle (degrees)",
        min_value=30.0,
        max_value=150.0,
        value=90.0,
        step=1.0,
        format="%.0f",
        key="L_angle"
    )
    
    l_poly_units = make_L_polygon(float(A), float(D), float(B), float(ang))
    
    # Preview
    prev = go.Figure()
    x = [p[0] for p in l_poly_units]
    y = [p[1] for p in l_poly_units]
    prev.add_trace(go.Scatter(
        x=x + [x[0]],
        y=y + [y[0]],
        mode="lines+markers",
        name="L-Shape",
        fill="toself"
    ))
    prev.update_yaxes(scaleanchor="x", scaleratio=1)
    prev.update_layout(
        title="L-Shape Preview",
        width=520,
        height=380,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(prev, use_container_width=True)
    
    with st.expander("➕ Add This L-Shape", expanded=True):
        qtyL = st.number_input(
            "Quantity",
            min_value=1,
            step=1,
            value=1,
            key="L_qty"
        )
        labelL = st.text_input("Label (optional)", value="", key="L_label")
        allow_rotL = st.checkbox(
            "Allow rotation for this part (0/90°)",
            value=True,
            key="L_rot"
        )
        
        add_cutL = st.checkbox("Add cutout(s) to this L-shape", key="L_cut_toggle")
        cut_listL: List[Tuple[float, float, float, float]] = []
        
        if add_cutL:
            ncuts = st.number_input(
                "Number of cutouts",
                min_value=1,
                max_value=5,
                value=1,
                step=1,
                key="L_cut_count"
            )
            for i in range(ncuts):
                st.write(f"**Cutout #{i+1}**")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    cw = st.number_input(
                        f"W{i+1}",
                        min_value=0.1,
                        value=30.0,
                        step=0.1,
                        format="%.2f",
                        key=f"L_cw_{i}"
                    )
                with c2:
                    ch = st.number_input(
                        f"H{i+1}",
                        min_value=0.1,
                        value=20.0,
                        step=0.1,
                        format="%.2f",
                        key=f"L_ch_{i}"
                    )
                with c3:
                    cx = st.number_input(
                        f"X{i+1}",
                        value=10.0,
                        step=0.5,
                        format="%.2f",
                        key=f"L_cx_{i}"
                    )
                with c4:
                    cy = st.number_input(
                        f"Y{i+1}",
                        value=10.0,
                        step=0.5,
                        format="%.2f",
                        key=f"L_cy_{i}"
                    )
                cut_listL.append((cx, cy, cw, ch))
        
        if st.button("➕ Add L-Shape", type="secondary", key="btn_add_L"):
            meta = {
                "is_L": True,
                "A": float(A),
                "B": float(B),
                "tA": float(D),
                "tB": float(D),
                "angle": float(ang)
            }
            
            if add_cutL and cut_listL:
                poly = polygon_with_cutouts(l_poly_units, cut_listL)
                meta["cutouts"] = cut_listL
                pts_out = list(poly.exterior.coords)
            else:
                pts_out = l_poly_units
            
            new_part = Part(
                id=str(uuid.uuid4()),
                label=labelL or f"L-{len(st.session_state.parts)+1}",
                qty=int(qtyL),
                shape_type="polygon",
                width=None,
                height=None,
                points=pts_out,
                allow_rotation=bool(allow_rotL),
                meta=meta
            )
            
            config = NestingConfig.from_session_state()
            is_valid, msg = validate_part(new_part, config.sheet_w, config.sheet_h)
            
            if is_valid:
                st.session_state.parts = st.session_state.parts + [new_part]
                st.session_state.needs_nest = True
                save_to_history()
                st.success(f"✅ Added L-shape{' (with cutouts)' if (add_cutL and cut_listL) else ''}")
                st.rerun()
            else:
                st.error(f"❌ {msg}")


def render_parts_editor(precision: int):
    """Render parts list editor"""
    st.subheader("📋 Parts List")
    
    if not st.session_state.parts:
        st.info("ℹ️ No parts yet. Add a shape above.")
        return
    
    rows = []
    for p in st.session_state.parts:
        w, h = p.get_dimensions()
        rows.append({
            "id": p.id,
            "Label": p.label,
            "Type": "L" if p.meta.get("is_L") else p.shape_type,
            "Width": round(w, precision) if w else None,
            "Height": round(h, precision) if h else None,
            "Qty": p.qty,
            "Allow Rotation": p.allow_rotation,
        })
    
    df = pd.DataFrame(rows)
    
    edited = st.data_editor(
        df.drop(columns=["id"]),
        use_container_width=True,
        num_rows="fixed",
        key="parts_editor"
    )
    
    if len(edited) == len(st.session_state.parts):
        changes_made = False
        for i, p in enumerate(st.session_state.parts):
            row = edited.iloc[i]
            
            if p.label != str(row["Label"]):
                p.label = str(row["Label"])
                changes_made = True
            
            if p.allow_rotation != bool(row["Allow Rotation"]):
                p.allow_rotation = bool(row["Allow Rotation"])
                changes_made = True
            
            if p.qty != int(row["Qty"]):
                p.qty = int(row["Qty"])
                changes_made = True
            
            if p.shape_type == "rect":
                w = float(row["Width"]) if row["Width"] is not None else p.width
                h = float(row["Height"]) if row["Height"] is not None else p.height
                if (p.width != w) or (p.height != h):
                    p.width = max(0.0, w or 0.0)
                    p.height = max(0.0, h or 0.0)
                    changes_made = True
                    st.session_state.needs_nest = True
        
        if changes_made:
            save_to_history()


def render_nesting_section():
    """Render nesting controls and preview"""
    st.markdown("---")
    st.header("🧩 Nesting")
    
    config = NestingConfig.from_session_state()
    mode = st.session_state.get("sb_mode", "Precision (polygon-aware)")
    
    # Nesting button with progress
    do_nest = st.button("🧩 Nest Parts", type="primary", key="btn_nest", use_container_width=True)
    
    if do_nest:
        if not st.session_state.parts:
            st.warning("⚠️ Add some parts first.")
        else:
            st.session_state.messages = []
            
            with st.spinner("🔄 Nesting parts... This may take a moment."):
                try:
                    if mode == "Fast (rectpack)":
                        sheets, util = rectpack_nest(st.session_state.parts, config)
                        st.session_state.placements = sheets
                        st.session_state.utilization = util
                    else:
                        sheets = list(poly_nest(st.session_state.parts, config))
                        placed_area = sum(
                            p["w"] * p["h"]
                            for s in sheets
                            for p in s["placements"]
                        )
                        total_area = len(sheets) * config.sheet_w * config.sheet_h
                        util = placed_area / max(1e-9, total_area)
                        st.session_state.placements = sheets
                        st.session_state.utilization = util
                    
                    st.session_state.needs_nest = False
                    save_to_history()
                    st.success(f"✅ Nesting complete! Used {len(sheets)} sheet(s)")
                    st.rerun()
                    
                except NestingError as e:
                    st.error(f"❌ Nesting failed: {e}")
                    st.info("💡 Try: reducing part sizes, increasing sheet size, or adjusting rotation settings")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {e}")
                    with st.expander("🐛 Debug Information"):
                        st.exception(e)
    
    # Display messages
    for m in st.session_state.get("messages", []):
        st.warning(m)
    
    # Display results
    if st.session_state.placements:
        render_nesting_results(config)


def render_nesting_results(config: NestingConfig):
    """Render nesting results and previews"""
    util_pct = round(st.session_state.utilization * 100, 2)
    used_area = sum(
        p["w"] * p["h"]
        for s in st.session_state.placements
        for p in s["placements"]
    )
    total_area = max(1e-9, len(st.session_state.placements) * config.sheet_w * config.sheet_h)
    
    # Metrics dashboard
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Sheets Used", len(st.session_state.placements))
    with col2:
        st.metric("📊 Utilization", f"{util_pct}%", delta=f"{util_pct - 75:.1f}%" if util_pct > 0 else None)
    with col3:
        st.metric("📐 Used Area", f"{round(used_area, 2)} {_pretty_units(config.units)}²")
    with col4:
        st.metric("📏 Total Area", f"{round(total_area, 2)} {_pretty_units(config.units)}²")
    
    if st.session_state.needs_nest:
        st.info("ℹ️ Parts have changed. Results are stale — click **Nest Parts** again.")
    
    st.markdown("---")
    st.subheader("📊 Sheet Previews (zoom & pan)")
    
    for si, sheet in enumerate(st.session_state.placements, start=1):
        W, H = sheet["sheet_w"], sheet["sheet_h"]
        fig = go.Figure()
        
        # Sheet boundary
        fig.add_shape(
            type="rect",
            x0=0,
            y0=0,
            x1=W,
            y1=H,
            line=dict(width=2, color="#333"),
            fillcolor="rgba(240, 240, 240, 0.1)"
        )
        
        # Parts
        colors = [
            "rgba(100, 150, 250, 0.15)",
            "rgba(250, 150, 100, 0.15)",
            "rgba(150, 250, 100, 0.15)",
            "rgba(250, 100, 150, 0.15)",
            "rgba(100, 250, 150, 0.15)",
        ]
        
        for idx, place in enumerate(sheet["placements"]):
            x, y, w, h = place["x"], place["y"], place["w"], place["h"]
            label = place["label"]
            color = colors[idx % len(colors)]
            
            # Part rectangle
            fig.add_shape(
                type="rect",
                x0=x,
                y0=y,
                x1=x + w,
                y1=y + h,
                line=dict(width=1, color="#1f77b4"),
                fillcolor=color
            )
            
            # Label
            fig.add_annotation(
                x=x + w / 2,
                y=y + h / 2,
                text=f"{label}<br>{round(w, config.precision)} × {round(h, config.precision)}",
                showarrow=False,
                font=dict(size=10, color="#333"),
                bgcolor="rgba(255, 255, 255, 0.7)",
                borderpad=4
            )
        
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(
            title=f"Sheet {si} of {len(st.session_state.placements)}",
            width=1000,
            height=700,
            margin=dict(l=30, r=30, t=50, b=30),
            xaxis=dict(
                range=[-1, W + 1],
                title=f"Width ({_pretty_units(config.units)})"
            ),
            yaxis=dict(
                range=[H + 1, -1],
                title=f"Height ({_pretty_units(config.units)})"
            ),
            dragmode="pan",
        )
        
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application entry point"""
    st.title("🔲 Nesting Tool Pro")
    st.caption("Advanced nesting tool with polygon support, auto-splitting, and L-shapes")
    
    # Load configuration
    config = NestingConfig.from_session_state()
    
    # Render sidebar
    render_sidebar(config)
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["✏️ Draw & Add", "📋 Parts List", "🧩 Nesting & Results"])
    
    with tab1:
        st.markdown("## ✏️ Design Your Parts")
        
        # Get tool and canvas
        tool, canvas_result, last_obj, px_per_unit = render_canvas_and_tools(
            st.session_state.get("sb_ppu", 8),
            st.session_state.get("sb_canvas_w", 1400),
            st.session_state.get("sb_canvas_h", 950),
            config.precision,
            config.units
        )
        
        # Tool-specific controls
        if tool in ["Rectangle", "Polygon (freehand)"]:
            render_add_shape_controls(
                tool,
                last_obj,
                canvas_result,
                px_per_unit,
                config.precision,
                config.units
            )
        elif tool == "L-shape (parametric)":
            render_L_shape_tool(config.precision, config.units)
    
    with tab2:
        st.markdown("## 📋 Manage Your Parts")
        render_parts_editor(config.precision)
        
        # Summary statistics
        if st.session_state.parts:
            st.markdown("---")
            st.markdown("### 📊 Summary")
            
            total_parts = len(st.session_state.parts)
            total_qty = sum(p.qty for p in st.session_state.parts)
            total_area = sum(
                p.get_dimensions()[0] * p.get_dimensions()[1] * p.qty
                for p in st.session_state.parts
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Unique Parts", total_parts)
            with col2:
                st.metric("Total Pieces", total_qty)
            with col3:
                st.metric(
                    "Total Part Area",
                    f"{round(total_area, 2)} {_pretty_units(config.units)}²"
                )
            
            # Show cutting list preview
            with st.expander("📋 View Cutting List"):
                cutting_list = export_cutting_list()
                st.dataframe(cutting_list, use_container_width=True)
    
    with tab3:
        st.markdown("## 🧩 Nest & Optimize")
        render_nesting_section()
    
    # Keyboard shortcuts info (footer)
    st.markdown("---")
    with st.expander("⌨️ Keyboard Shortcuts & Tips"):
        st.markdown("""
        **Shortcuts:**
        - `Ctrl+Z` / `↶ Undo`: Undo last action
        - `Ctrl+Y` / `↷ Redo`: Redo last undone action
        
        **Tips:**
        - Draw multiple shapes on the canvas, then use **Add All Drawn Shapes** to batch add
        - Use the data editor in the Parts List tab to quickly update multiple parts
        - Export your project regularly to save your work
        - The cutting list CSV can be imported into spreadsheet software
        - Larger clearance values help prevent parts from being too close together
        - Auto-split is useful for very large parts that exceed sheet dimensions
        """)


if __name__ == "__main__":
    main()
