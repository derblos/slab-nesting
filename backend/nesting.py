"""
Nesting algorithms extracted from the main application

This module contains the core nesting logic that can be used
by both the Streamlit UI and the FastAPI backend.
"""

import math
import uuid
from typing import List, Tuple, Any, Dict
from rectpack import newPacker
from shapely.geometry import Polygon, box as shp_box
from shapely.affinity import rotate as shp_rotate, translate as shp_translate

# Import from local models (will be converted to use backend.models)
try:
    from .models import Part, NestingConfig
except ImportError:
    # Fallback for when run from app.py context
    from backend.models import Part, NestingConfig


def split_rect_if_needed(p: Part, config: NestingConfig) -> Tuple[List[Part], bool]:
    """
    Split rectangle if it exceeds sheet dimensions

    Returns: (list of parts, was_split_boolean)
    """
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


def _rectpack_trial(parts_trial: List[Part], sheet_w: float, sheet_h: float,
                     clearance: float, rotation: bool) -> Tuple[List[Any], None]:
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


def rectpack_nest(parts: List[Part], config: NestingConfig) -> Tuple[List[Dict], float]:
    """
    Fast rectangular nesting using rectpack

    Returns: (sheets, utilization)
    """
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


def calculate_total_area(sheets: List[Dict], config: NestingConfig) -> Tuple[float, float]:
    """
    Calculate used and available area

    Returns: (used_area, total_area)
    """
    used_area = sum(
        p["w"] * p["h"]
        for s in sheets
        for p in s.get("placements", [])
    )
    total_area = len(sheets) * config.sheet_w * config.sheet_h if sheets else 0.0
    return used_area, total_area
