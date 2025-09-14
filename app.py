# app.py — Polygon-aware nesting + L-split policy (Streamlit)
from __future__ import annotations

import json, math, uuid
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

import pandas as pd
import plotly.graph_objects as go
from rectpack import newPacker
from shapely.affinity import rotate as shp_rotate, translate as shp_translate
from shapely.geometry import Polygon, box as shp_box
from shapely.ops import unary_union

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import streamlit.components.v1 as components

st.set_page_config(page_title="Nesting Tool", layout="wide")
SHOW_CSV = False  # draw-only workflow

# ───────────────────────── Data model / state ─────────────────────────
@dataclass
class Part:
    id: str
    label: str
    qty: int
    shape_type: str            # "rect" or "polygon"
    width: float | None
    height: float | None
    points: List[Tuple[float, float]] | None
    allow_rotation: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)

def _init_state():
    st.session_state.setdefault("parts", [])
    st.session_state.setdefault("needs_nest", True)
    st.session_state.setdefault("placements", [])
    st.session_state.setdefault("utilization", 0.0)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("_open_editor", False)

_init_state()

def _pretty_units(u: str) -> str:
    return {"in": "in", "mm": "mm", "cm": "cm"}[u]

# ───────────────────────── Fabric helpers ─────────────────────────
def _fabric_rect_dims(obj: Dict) -> Tuple[float, float]:
    w = float(obj.get("width", 0)) * float(obj.get("scaleX", 1.0))
    h = float(obj.get("height", 0)) * float(obj.get("scaleY", 1.0))
    return abs(w), abs(h)

def _fabric_polygon_points(obj: Dict) -> List[Tuple[float, float]]:
    pts = []
    path = obj.get("path")
    if isinstance(path, list):
        x0 = float(obj.get("left", 0)); y0 = float(obj.get("top", 0))
        sx = float(obj.get("scaleX", 1.0)); sy = float(obj.get("scaleY", 1.0))
        for seg in path:
            if isinstance(seg, list) and len(seg) >= 3 and seg[0] in ("L", "M"):
                px = x0 + float(seg[1]) * sx
                py = y0 + float(seg[2]) * sy
                pts.append((px, py))
    return pts

def _bbox_of_polygon(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    return (max(xs) - min(xs), max(ys) - min(ys))

# ───────────────────────── Sidebar ─────────────────────────
with st.sidebar:
    st.header("Project settings")
    units = st.selectbox("Units", ["in", "mm", "cm"], index=0)
    precision = st.number_input("Precision (decimals)", 0, 4, 2)
    px_per_unit = st.slider("Canvas scale (px per unit)", 2, 20, 8)

    st.markdown("### Sheet")
    sheet_w = st.number_input(f"Sheet width ({_pretty_units(units)})", min_value=1.0, value=97.0, step=1.0, format="%.2f")
    sheet_h = st.number_input(f"Sheet height ({_pretty_units(units)})", min_value=1.0, value=80.50, step=1.0, format="%.2f")
    clearance = st.number_input(f"Clearance between parts ({_pretty_units(units)})", min_value=0.0, value=0.25, step=0.05, format="%.2f")
    allow_rotation_global = st.checkbox("Allow rotation globally (0/90°)", value=True)

    st.markdown("### Nesting mode")
    mode = st.radio("Mode", ["Fast (rectpack)", "Precision (polygon-aware)"], index=0, help="Precision places true outlines; slower but more accurate for L-shapes.")

    st.markdown("### Auto-split oversized rectangles")
    autosplit_rects = st.checkbox("Auto-split rectangles that exceed sheet size", value=True)
    seam_gap = st.number_input(f"Seam/kerf gap at split ({_pretty_units(units)})", min_value=0.0, value=0.125, step=0.01, format="%.3f")
    min_leg = st.number_input(f"Minimum leg length after split ({_pretty_units(units)})", min_value=1.0, value=6.0, step=0.5, format="%.2f")
    prefer_long_split = st.selectbox("Prefer split along", ["long side", "short side"], index=0)

    st.markdown("### L-shape split policy")
    enable_L_seams = st.checkbox("Allow seams on L-shapes when needed", value=True)
    L_max_leg_no_split = st.number_input(f"Max leg length without split ({_pretty_units(units)})", min_value=1.0, value=48.0, step=1.0, format="%.0f")
    # Depth equality assumption: tA == tB == D (can relax later)

    st.markdown("---")
    if st.button("🗑️ Clear project", use_container_width=True, type="primary"):
        for k in ["parts", "needs_nest", "placements", "utilization", "messages", "_open_editor"]:
            st.session_state.pop(k, None)
        _init_state()
        st.success("Project cleared.")

# ───────────────────────── Main layout ─────────────────────────
left, right = st.columns([0.55, 0.45])

with left:
    st.title("Nesting Tool")

    # Tool chooser
    tool = st.radio(
        "Tool",
        ["Rectangle", "Polygon (freehand)", "L-shape (parametric)"],
        horizontal=False
    )

    # Drawing canvas for rect/polygon
    canvas_w, canvas_h = 900, 520
    last_obj = None; live_w = live_h = None

    if tool in ["Rectangle", "Polygon (freehand)"]:
        drawing_mode = "rect" if tool == "Rectangle" else "polygon"
        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0.0)",
            stroke_width=2,
            stroke_color="#1f77b4",
            background_color="#fafafa",
            height=canvas_h,
            width=canvas_w,
            drawing_mode=drawing_mode,
            display_toolbar=True,
            update_streamlit=True,
            key="draw_canvas"
        )
        if canvas_result and canvas_result.json_data:
            objs = canvas_result.json_data.get("objects", [])
            if objs:
                last_obj = objs[-1]
                if drawing_mode == "rect" and last_obj.get("type") == "rect":
                    w_px, h_px = _fabric_rect_dims(last_obj)
                    live_w = round(w_px / px_per_unit, precision)
                    live_h = round(h_px / px_per_unit, precision)
                elif drawing_mode != "rect" and last_obj.get("path"):
                    pts = _fabric_polygon_points(last_obj)
                    if len(pts) >= 3:
                        bw, bh = _bbox_of_polygon(pts)
                        live_w = round(bw / px_per_unit, precision)
                        live_h = round(bh / px_per_unit, precision)
        st.caption("**Live dimensions**: " + (f"{live_w} × {live_h} {_pretty_units(units)}" if live_w is not None else "— start/continue drawing —"))

    # Parametric L-shape
    l_poly_units = None; A=B=tA=tB=angle=None
    if tool == "L-shape (parametric)":
        st.write("Define outer legs A & B, depth D (tA=tB), and angle (defaults 90°).")
        colA, colB = st.columns(2)
        with colA:
            A = st.number_input(f"Outer leg A length ({_pretty_units(units)})", min_value=1.0, value=120.0, step=0.5, format="%.2f")
            tA = st.number_input(f"Depth / thickness D ({_pretty_units(units)})", min_value=0.1, value=25.0, step=0.1, format="%.2f")
        with colB:
            B = st.number_input(f"Outer leg B length ({_pretty_units(units)})", min_value=1.0, value=60.0, step=0.5, format="%.2f")
            tB = tA  # assume equal depths for now
        angle = st.number_input("Inside angle (degrees)", min_value=30.0, max_value=150.0, value=90.0, step=1.0, format="%.0f")

        def make_L_polygon(A, D, B, angle_deg) -> List[Tuple[float, float]]:
            # Right-angle base L outline; rotate if angle != 90
            pts = [(0,0),(A,0),(A,D),(D,D),(D,B),(0,B),(0,0)]
            if abs(angle_deg - 90.0) > 1e-6:
                theta = math.radians(90.0 - angle_deg)
                rot = lambda p: (p[0]*math.cos(theta)-p[1]*math.sin(theta), p[0]*math.sin(theta)+p[1]*math.cos(theta))
                pts = [rot(p) for p in pts]
            return pts

        l_poly_units = make_L_polygon(float(A), float(tA), float(B), float(angle))
        prev = go.Figure()
        x = [p[0] for p in l_poly_units]; y = [p[1] for p in l_poly_units]
        prev.add_trace(go.Scatter(x=x+[x[0]], y=y+[y[0]], mode="lines+markers", name="L"))
        prev.update_yaxes(scaleanchor="x", scaleratio=1)
        prev.update_layout(title="L preview", width=480, height=360, margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(prev, use_container_width=True)

    # Add current shape
    with st.expander("Add the current shape to the Parts list", expanded=True):
        default_qty = st.number_input("Quantity", min_value=1, step=1, value=1)
        default_label = st.text_input("Label (optional)", value="")
        allow_rot = st.checkbox("Allow rotation for this part (0/90°)", value=True)
        if st.button("➕ Add this shape", type="secondary"):
            if tool == "Rectangle":
                if not last_obj or last_obj.get("type") != "rect":
                    st.warning("Draw a rectangle first.")
                else:
                    w_px, h_px = _fabric_rect_dims(last_obj)
                    w = round(w_px / px_per_unit, precision); h = round(h_px / px_per_unit, precision)
                    if w <= 0 or h <= 0:
                        st.error("This rectangle has zero width/height.")
                    else:
                        st.session_state.parts.append(Part(
                            id=str(uuid.uuid4()), label=default_label or f"Draw-{len(st.session_state.parts)+1}",
                            qty=int(default_qty), shape_type="rect",
                            width=float(w), height=float(h), points=None,
                            allow_rotation=bool(allow_rot)
                        ))
                        st.session_state.needs_nest = True
                        st.success(f"Added rectangle ({w} × {h} {_pretty_units(units)})")

            elif tool == "Polygon (freehand)":
                if not last_obj or not last_obj.get("path"):
                    st.warning("Draw a polygon first.")
                else:
                    pts = _fabric_polygon_points(last_obj)
                    if len(pts) < 3:
                        st.error("Polygon needs at least 3 points.")
                    else:
                        pts_units = [(x/px_per_unit, y/px_per_unit) for (x,y) in pts]
                        st.session_state.parts.append(Part(
                            id=str(uuid.uuid4()), label=default_label or f"Draw-{len(st.session_state.parts)+1}",
                            qty=int(default_qty), shape_type="polygon",
                            width=None, height=None, points=pts_units,
                            allow_rotation=bool(allow_rot)
                        ))
                        st.session_state.needs_nest = True
                        st.success("Added polygon.")

            else:  # L-shape (parametric)
                if not l_poly_units or len(l_poly_units) < 3:
                    st.error("L-shape parameters invalid.")
                else:
                    st.session_state.parts.append(Part(
                        id=str(uuid.uuid4()), label=default_label or f"L-{len(st.session_state.parts)+1}",
                        qty=int(default_qty), shape_type="polygon",
                        width=None, height=None, points=l_poly_units,
                        allow_rotation=bool(allow_rot),
                        meta={"is_L": True, "A": float(A), "B": float(B), "tA": float(tA), "tB": float(tB), "angle": float(angle)}
                    ))
                    st.session_state.needs_nest = True
                    st.success("Added L-shape.")

    # Parts editor
    st.markdown("---")
    st.subheader("Parts")
    if not st.session_state.parts:
        st.info("No parts yet. Add a shape above.")
    else:
        rows = []
        for p in st.session_state.parts:
            rows.append({
                "id": p.id, "Label": p.label,
                "Type": ("L" if p.meta.get("is_L") else p.shape_type),
                "Width": round(p.width, precision) if p.width is not None else None,
                "Height": round(p.height, precision) if p.height is not None else None,
                "Qty": p.qty,
                "Allow Rotation": p.allow_rotation,
            })
        df = pd.DataFrame(rows)
        edited = st.data_editor(df.drop(columns=["id"]), use_container_width=True, num_rows="fixed", key="parts_editor")
        if len(edited) == len(st.session_state.parts):
            for i, p in enumerate(st.session_state.parts):
                row = edited.iloc[i]
                p.label = str(row["Label"]); p.allow_rotation = bool(row["Allow Rotation"]); p.qty = int(row["Qty"])
                if p.shape_type == "rect":
                    w = float(row["Width"]) if row["Width"] is not None else p.width
                    h = float(row["Height"]) if row["Height"] is not None else p.height
                    if (p.width != w) or (p.height != h):
                        p.width = max(0.0, w or 0.0); p.height = max(0.0, h or 0.0)
                        st.session_state.needs_nest = True

with right:
    st.subheader("Nesting")

    # ───────────────────────── Split helpers ─────────────────────────
    def split_rect_if_needed(p: Part, sheet_w: float, sheet_h: float,
                             min_leg: float, seam_gap: float, prefer_long: bool, clearance: float) -> tuple[list[Part], bool]:
        """Split an oversized rectangle into panels that fit; returns (subs, did_split)."""
        if p.shape_type != "rect" or p.width is None or p.height is None:
            return [p], False
        W, H = p.width, p.height
        usable_w = max(0.0, sheet_w - clearance)
        usable_h = max(0.0, sheet_h - clearance)
        if W <= usable_w and H <= usable_h:
            return [p], False

        parts_out: List[Part] = []
        long_side = "W" if W >= H else "H"
        split_along_width = (long_side == "W") if prefer_long else (long_side == "H")
        if W > usable_w and H <= usable_h: split_along_width = True
        if H > usable_h and W <= usable_w: split_along_width = False

        if split_along_width:
            remaining = W; idx = 1
            while remaining > 1e-9:
                panel = min(usable_w, remaining)
                if panel < min_leg - 1e-9:
                    if parts_out:
                        prev = parts_out.pop()
                        merged = prev.width + seam_gap + panel
                        if merged > usable_w + 1e-9:
                            st.session_state.messages.append(f"⚠️ Cannot split {p.label}: panel too short.")
                            return [p], False
                        parts_out.append(Part(id=str(uuid.uuid4()), label=f"{p.label}-S{idx-1}", qty=p.qty,
                                              shape_type="rect", width=merged, height=H, points=None,
                                              allow_rotation=p.allow_rotation, meta={"from": p.id}))
                    else:
                        st.session_state.messages.append(f"⚠️ Cannot split {p.label}: min leg violated.")
                        return [p], False
                else:
                    parts_out.append(Part(id=str(uuid.uuid4()), label=f"{p.label}-S{idx}", qty=p.qty,
                                          shape_type="rect", width=panel, height=H, points=None,
                                          allow_rotation=p.allow_rotation, meta={"from": p.id}))
                remaining -= panel
                if remaining > 1e-9: remaining -= seam_gap
                idx += 1
        else:
            remaining = H; idx = 1
            while remaining > 1e-9:
                panel = min(usable_h, remaining)
                if panel < min_leg - 1e-9:
                    if parts_out:
                        prev = parts_out.pop()
                        merged = prev.height + seam_gap + panel
                        if merged > usable_h + 1e-9:
                            st.session_state.messages.append(f"⚠️ Cannot split {p.label}: panel too short.")
                            return [p], False
                        parts_out.append(Part(id=str(uuid.uuid4()), label=f"{p.label}-S{idx-1}", qty=p.qty,
                                              shape_type="rect", width=W, height=merged, points=None,
                                              allow_rotation=p.allow_rotation, meta={"from": p.id}))
                    else:
                        st.session_state.messages.append(f"⚠️ Cannot split {p.label}: min leg violated.")
                        return [p], False
                else:
                    parts_out.append(Part(id=str(uuid.uuid4()), label=f"{p.label}-S{idx}", qty=p.qty,
                                          shape_type="rect", width=W, height=panel, points=None,
                                          allow_rotation=p.allow_rotation, meta={"from": p.id}))
                remaining -= panel
                if remaining > 1e-9: remaining -= seam_gap
                idx += 1

        return (parts_out if parts_out else [p]), (len(parts_out) > 1)

    def decompose_L_prefer_corner(p: Part) -> list[Part]:
        """Default corner split: Rect1=(A-D)×D (long leg), Rect2=B×D (short leg)."""
        m = p.meta or {}
        if not m.get("is_L"): return [p]
        if abs(float(m.get("angle", 90.0)) - 90.0) > 1e-6: return [p]
        A = float(m.get("A", 0.0)); B = float(m.get("B", 0.0)); D = float(m.get("tA", 0.0))
        r1 = Part(id=str(uuid.uuid4()), label=f"{p.label}-legA", qty=p.qty, shape_type="rect",
                  width=max(0.0, A - D), height=D, points=None, allow_rotation=p.allow_rotation, meta={"from_L": p.id, "leg":"A"})
        r2 = Part(id=str(uuid.uuid4()), label=f"{p.label}-legB", qty=p.qty, shape_type="rect",
                  width=D, height=B, points=None, allow_rotation=p.allow_rotation, meta={"from_L": p.id, "leg":"B"})
        if r1.width <= 0 or r1.height <= 0 or r2.width <= 0 or r2.height <= 0:
            return [p]
        return [r1, r2]

    def decompose_L_alternate(p: Part) -> list[Part]:
        """Alternate: Rect1=A×D, Rect2=(B−D)×D."""
        m = p.meta or {}
        if not m.get("is_L"): return [p]
        if abs(float(m.get("angle", 90.0)) - 90.0) > 1e-6: return [p]
        A = float(m.get("A", 0.0)); B = float(m.get("B", 0.0)); D = float(m.get("tA", 0.0))
        r1 = Part(id=str(uuid.uuid4()), label=f"{p.label}-legA", qty=p.qty, shape_type="rect",
                  width=A, height=D, points=None, allow_rotation=p.allow_rotation, meta={"from_L": p.id, "leg":"A"})
        r2 = Part(id=str(uuid.uuid4()), label=f"{p.label}-legB", qty=p.qty, shape_type="rect",
                  width=max(0.0, B - D), height=D, points=None, allow_rotation=p.allow_rotation, meta={"from_L": p.id, "leg":"B"})
        if r1.width <= 0 or r1.height <= 0 or r2.width <= 0 or r2.height <= 0:
            return [p]
        return [r1, r2]

    # ───────────────────────── Nesting engines ─────────────────────────
    def rectpack_nest(parts: List[Part], sheet_w: float, sheet_h: float, clearance: float, rotation: bool):
        """Fast rectangle nesting; polygons use bounding boxes."""
        to_add = []
        expanded: List[Part] = []
        any_split = False

        # Expand & split per rules
        for p in parts:
            if p.qty <= 0: continue
            subs = [p]
            did_split = False

            # L split policy
            if p.meta.get("is_L") and enable_L_seams and abs(p.meta.get("angle", 90.0) - 90.0) < 1e-6:
                A = float(p.meta.get("A", 0.0)); B = float(p.meta.get("B", 0.0)); D = float(p.meta.get("tA", 0.0))
                # No split if both legs <= threshold and it fits sheet
                if not (A <= L_max_leg_no_split and B <= L_max_leg_no_split and A <= sheet_w and B <= sheet_h):
                    # Try corner-first vs alternate and pick better (fewest sheets) using a tiny dry-run
                    cand1 = decompose_L_prefer_corner(p)  # [(A-D)xD, BxD]
                    cand2 = decompose_L_alternate(p)      # [A x D, (B-D)xD]
                    s1, _ = _rectpack_trial(cand1, sheet_w, sheet_h, clearance, rotation)
                    s2, _ = _rectpack_trial(cand2, sheet_w, sheet_h, clearance, rotation)
                    subs = cand1 if len(s1) <= len(s2) else cand2
                    did_split = True

            # Rectangle auto-split (secondary; honor "split on other end" by not re-splitting the other leg we just split)
            if autosplit_rects:
                next_subs = []
                for s in subs:
                    if s.shape_type == "rect":
                        ss, did = split_rect_if_needed(s, sheet_w, sheet_h, min_leg, seam_gap, prefer_long_split == "long side", clearance)
                        did_split = did_split or did
                        next_subs.extend(ss)
                    else:
                        next_subs.append(s)
                subs = next_subs

            any_split = any_split or did_split
            for s in subs:
                for _ in range(p.qty):
                    expanded.append(s)

        rotation_effective = rotation and (not any_split)
        packer = newPacker(rotation=rotation_effective)
        EPS = 1e-6

        for s in expanded:
            if s.shape_type == "rect" and s.width and s.height:
                w = min(s.width + clearance, sheet_w - EPS)
                h = min(s.height + clearance, sheet_h - EPS)
            else:
                # polygons -> pack by bbox
                if s.points:
                    poly = Polygon(s.points); minx, miny, maxx, maxy = poly.bounds
                    w = min((maxx - minx) + clearance, sheet_w - EPS)
                    h = min((maxy - miny) + clearance, sheet_h - EPS)
                else:
                    continue
            rid = f"{s.label}#{uuid.uuid4().hex[:4]}"
            to_add.append((w, h, rid, s.label))

        if not to_add:
            return [], 0.0

        for (w, h, rid, _) in to_add: packer.add_rect(w, h, rid=rid)
        for _ in range(200): packer.add_bin(sheet_w, sheet_h)
        packer.pack()

        sheets = []
        total_area = 0.0
        for abin in packer:
            rects = abin.rect_list()
            if not rects: continue
            placements = []
            for (x, y, w, h, rid) in rects:
                label = rid.split("#")[0]
                placements.append({"x": x, "y": y, "w": w - clearance, "h": h - clearance, "rid": rid, "label": label})
                total_area += max(0.0, (w - clearance)) * max(0.0, (h - clearance))
            sheets.append({"sheet_w": sheet_w, "sheet_h": sheet_h, "placements": placements})

        util = total_area / max(1e-9, len(sheets)*sheet_w*sheet_h)
        return sheets, util

    def _rectpack_trial(parts_trial, sheet_w, sheet_h, clearance, rotation):
        """Tiny helper: run rectpack quickly to compare splits."""
        packer = newPacker(rotation=rotation)
        EPS = 1e-6
        for s in parts_trial:
            w = min((s.width or 0) + clearance, sheet_w - EPS)
            h = min((s.height or 0) + clearance, sheet_h - EPS)
            packer.add_rect(w, h, rid=str(uuid.uuid4()))
        for _ in range(50): packer.add_bin(sheet_w, sheet_h)
        packer.pack()
        return [abin for abin in packer if abin.rect_list()], None

    def poly_nest(parts: List[Part], sheet_w: float, sheet_h: float, clearance: float, rotation: bool, grid_step: float = 0.5):
        """
        Polygon-aware greedy placer:
          - Converts each part instance to a Shapely polygon (rects -> rectangles; L/polygons -> true outline).
          - Tries (0°) and (90°) if rotation allowed.
          - Scans positions on a grid (grid_step) left-to-right, top-to-bottom.
          - Places if (within sheet) and (no intersection with placed polys buffered by clearance/2).
        """
        # Expand parts by qty and apply L split policy first (panelization when needed)
        expanded: List[Tuple[Part, Polygon]] = []
        any_split = False

        def part_to_poly(s: Part) -> Polygon | None:
            if s.shape_type == "rect" and s.width and s.height:
                return shp_box(0, 0, s.width, s.height)
            if s.shape_type == "polygon" and s.points:
                try:
                    return Polygon(s.points)
                except Exception:
                    return None
            return None

        for p in parts:
            if p.qty <= 0: continue
            subs = [p]; did_split = False

            if p.meta.get("is_L") and enable_L_seams and abs(p.meta.get("angle",90.0)-90.0) < 1e-6:
                A = float(p.meta.get("A",0.0)); B = float(p.meta.get("B",0.0)); D = float(p.meta.get("tA",0.0))
                # Don't split only if both legs <= threshold and L fits sheet
                L_poly = part_to_poly(p)
                fits_sheet = False
                if L_poly is not None:
                    minx, miny, maxx, maxy = L_poly.bounds
                    fits_sheet = (maxx - minx) <= sheet_w and (maxy - miny) <= sheet_h
                if not (A <= L_max_leg_no_split and B <= L_max_leg_no_split and fits_sheet):
                    cand1 = decompose_L_prefer_corner(p)
                    cand2 = decompose_L_alternate(p)
                    s1, _ = _rectpack_trial(cand1, sheet_w, sheet_h, clearance, rotation)
                    s2, _ = _rectpack_trial(cand2, sheet_w, sheet_h, clearance, rotation)
                    subs = cand1 if len(s1) <= len(s2) else cand2
                    did_split = True

            # Rectangle auto-split if still too big (applies “second split on other end” by splitting whichever leg still exceeds)
            if autosplit_rects:
                next_subs = []
                for s in subs:
                    if s.shape_type == "rect":
                        ss, did = split_rect_if_needed(s, sheet_w, sheet_h, min_leg, seam_gap, prefer_long_split == "long side", clearance)
                        did_split = did_split or did
                        next_subs.extend(ss)
                    else:
                        next_subs.append(s)
                subs = next_subs

            any_split = any_split or did_split

            for s in subs:
                shp = part_to_poly(s)
                if shp is None: continue
                for _ in range(s.qty):
                    expanded.append((s, shp))

        rotation_effective = rotation and (not any_split)

        # Placement
        placed = []  # list of dicts with polygon+meta
        sheet_poly = shp_box(0, 0, sheet_w, sheet_h)
        buffer_clear = clearance / 2.0 if clearance > 0 else 0.0

        for s, base_poly in sorted(expanded, key=lambda t: t[1].area, reverse=True):
            orientations = [0]
            if rotation_effective: orientations = [0, 90]
            placed_ok = False

            for ang in orientations:
                poly = shp_rotate(base_poly, ang, origin=(0,0), use_radians=False)
                # scan grid
                y = 0.0
                while y <= sheet_h and not placed_ok:
                    x = 0.0
                    while x <= sheet_w and not placed_ok:
                        test = shp_translate(poly, xoff=x, yoff=y)
                        # bounds check with clearance
                        bounds_ok = test.buffer(buffer_clear).within(sheet_poly)
                        if not bounds_ok:
                            x += grid_step; continue
                        # collision check
                        collision = False
                        for other in placed:
                            if test.buffer(buffer_clear).intersects(other["poly"].buffer(buffer_clear)):
                                collision = True; break
                        if not collision:
                            placed.append({"poly": test, "label": s.label, "rid": f"{s.label}#{uuid.uuid4().hex[:4]}", "angle": ang})
                            placed_ok = True
                            break
                        x += grid_step
                    y += grid_step
                if placed_ok: break

            if not placed_ok:
                # start a new sheet: finalize current and start placement anew
                # collect current sheet placements
                yield {"sheet_w": sheet_w, "sheet_h": sheet_h, "placements": [
                    _poly_to_rect_anno(d["poly"], d["label"], precision) for d in placed
                ]}
                placed = []
                # place this part on the new sheet at (0,0) if possible
                poly0 = shp_rotate(base_poly, 0, origin=(0,0), use_radians=False)
                if poly0.buffer(buffer_clear).within(sheet_poly):
                    placed.append({"poly": poly0, "label": s.label, "rid": f"{s.label}#{uuid.uuid4().hex[:4]}", "angle": 0})
                else:
                    # try 90 if allowed
                    poly90 = shp_rotate(base_poly, 90, origin=(0,0), use_radians=False)
                    if rotation_effective and poly90.buffer(buffer_clear).within(sheet_poly):
                        placed.append({"poly": poly90, "label": s.label, "rid": f"{s.label}#{uuid.uuid4().hex[:4]}", "angle": 90})
                    else:
                        st.session_state.messages.append(f"⚠️ {s.label}: cannot place on empty sheet (too large).")
        # flush last sheet
        if placed:
            yield {"sheet_w": sheet_w, "sheet_h": sheet_h, "placements": [
                _poly_to_rect_anno(d["poly"], d["label"], precision) for d in placed
            ]}

    def _poly_to_rect_anno(poly: Polygon, label: str, precision: int):
        """Represent placed polygon by its bbox for preview labels; keep exact dims in text."""
        minx, miny, maxx, maxy = poly.bounds
        w = maxx - minx; h = maxy - miny
        return {"x": float(minx), "y": float(miny), "w": float(w), "h": float(h), "rid": f"{label}#{uuid.uuid4().hex[:4]}", "label": label,
                "exact_w": round(w, precision), "exact_h": round(h, precision)}

    # ───────────────────────── Run nesting ─────────────────────────
    do_nest = st.button("🧩 Nest parts", type="primary", use_container_width=True)
    if do_nest:
        if not st.session_state.parts:
            st.warning("Add some parts first.")
        else:
            st.session_state.messages = []
            if mode == "Fast (rectpack)":
                sheets, util = rectpack_nest(st.session_state.parts, sheet_w, sheet_h, clearance, allow_rotation_global)
                st.session_state.placements, st.session_state.utilization = sheets, util
            else:
                # Precision mode returns a generator of sheets
                sheets = list(poly_nest(st.session_state.parts, sheet_w, sheet_h, clearance, allow_rotation_global, grid_step=0.5))
                # utilization by true polygon area
                placed_area = 0.0
                for s in sheets:
                    for p in s["placements"]:
                        placed_area += p["w"] * p["h"]  # bbox area as a lower bound
                util = placed_area / max(1e-9, len(sheets)*sheet_w*sheet_h)
                st.session_state.placements, st.session_state.utilization = sheets, util
            st.session_state.needs_nest = False
            st.session_state._open_editor = False

    # Messages
    if st.session_state.messages:
        for m in st.session_state.messages:
            st.warning(m)

    # ───────────────────────── Preview ─────────────────────────
    if st.session_state.placements:
        util_pct = round(st.session_state.utilization * 100, 2)
        used_area = sum(p["w"] * p["h"] for s in st.session_state.placements for p in s["placements"])
        total_area = max(1e-9, len(st.session_state.placements) * sheet_w * sheet_h)

        st.markdown(f"**Sheets used:** {len(st.session_state.placements)}")
        st.markdown(f"**Utilization (approx):** {util_pct}% (used {round(used_area,2)} / {round(total_area,2)} {_pretty_units(units)}²)")
        if st.session_state.needs_nest:
            st.info("Parts changed. Results are stale — click **Nest parts** again.")

        st.markdown("---")
        st.subheader("Sheet preview (zoom & pan)")
        for si, sheet in enumerate(st.session_state.placements, start=1):
            W, H = sheet["sheet_w"], sheet["sheet_h"]
            fig = go.Figure()
            fig.add_shape(type="rect", x0=0, y0=0, x1=W, y1=H, line=dict(width=2), fillcolor="rgba(240,240,240,0.1)")
            for place in sheet["placements"]:
                x, y, w, h, label = place["x"], place["y"], place["w"], place["h"], place["label"]
                fig.add_shape(type="rect", x0=x, y0=y, x1=x+w, y1=y+h, line=dict(width=1), fillcolor="rgba(100,150,250,0.15)")
                fig.add_annotation(x=x+w/2, y=y+h/2,
                                   text=f"{label}<br>{round(w, precision)} × {round(h, precision)}",
                                   showarrow=False, font=dict(size=12))
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            fig.update_layout(
                title=f"Sheet {si}", width=900, height=650, margin=dict(l=30, r=30, t=50, b=30),
                xaxis=dict(range=[-1, W+1], title=f"Width ({_pretty_units(units)})"),
                yaxis=dict(range=[H+1, -1], title=f"Height ({_pretty_units(units)})"),
                dragmode="pan",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No nesting results yet. Click **Nest parts** when you’re ready.")
