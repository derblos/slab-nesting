# app.py  — Draw-only nesting with auto-split + parametric L-shape
from __future__ import annotations

import math, uuid
from dataclasses import dataclass
from typing import List, Tuple, Dict

import pandas as pd
import plotly.graph_objects as go
from rectpack import newPacker
from shapely.geometry import Polygon

import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Nesting Tool", layout="wide")
SHOW_CSV = False  # keep False for draw-only workflow

# ───────────────────────── Models / State ─────────────────────────
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

def _init_state():
    st.session_state.setdefault("parts", [])
    st.session_state.setdefault("needs_nest", True)
    st.session_state.setdefault("placements", [])
    st.session_state.setdefault("utilization", 0.0)
    st.session_state.setdefault("messages", [])

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
    allow_rotation_global = st.checkbox("Allow rotation globally", value=True)

    st.markdown("### Auto-split oversized rectangles")
    autosplit = st.checkbox("Auto-split rectangles that exceed sheet size", value=True)
    seam_gap = st.number_input(f"Seam/kerf gap at split ({_pretty_units(units)})", min_value=0.0, value=0.125, step=0.01, format="%.3f")
    min_leg = st.number_input(f"Minimum leg length after split ({_pretty_units(units)})", min_value=1.0, value=6.0, step=0.5, format="%.2f")
    prefer_long_split = st.selectbox("Prefer split along", ["long side", "short side"], index=0)

    st.markdown("---")
    if st.button("🗑️ Clear project", use_container_width=True, type="primary"):
        for k in ["parts", "needs_nest", "placements", "utilization", "messages"]:
            st.session_state.pop(k, None)
        _init_state()
        st.success("Project cleared.")

# ───────────────────────── Main layout ─────────────────────────
left, right = st.columns([0.55, 0.45])

with left:
    st.title("Nesting Tool — Draw Only")

    # ------- Tool chooser (added L-shape) -------
    tool = st.radio("Tool", ["Rectangle", "Polygon (freehand)", "L-shape (parametric)"], horizontal=False)

    # ------- Drawing canvas for rect/polygon -------
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

    # ------- Parametric L-shape input -------
    l_poly_units = None
    if tool == "L-shape (parametric)":
        st.write("Define outer legs (A and B), inside notch, and angle. Defaults create a right-angle L; change angle for a mitered/angled inside corner.")
        colA, colB = st.columns(2)
        with colA:
            A = st.number_input(f"Outer leg A length ({_pretty_units(units)})", min_value=1.0, value=48.0, step=0.5, format="%.2f")
            tA = st.number_input(f"Leg A width/thickness ({_pretty_units(units)})", min_value=1.0, value=26.0, step=0.5, format="%.2f")
        with colB:
            B = st.number_input(f"Outer leg B length ({_pretty_units(units)})", min_value=1.0, value=30.0, step=0.5, format="%.2f")
            tB = st.number_input(f"Leg B width/thickness ({_pretty_units(units)})", min_value=1.0, value=26.0, step=0.5, format="%.2f")
        angle = st.number_input("Inside angle (degrees)", min_value=30.0, max_value=150.0, value=90.0, step=1.0, format="%.0f")

        # Build polygon for L: start at origin, go around perimeter (simple 8-point L)
        # For angle != 90°, rotate the B leg around the inside corner.
        def make_L_polygon(A, tA, B, tB, angle_deg) -> List[Tuple[float, float]]:
            # Base right-angle L in local coords (origin at inside corner)
            # Outline order: along A leg, around outside, back along B leg, around inside notch, return.
            pts = [
                (0, 0),
                (A, 0),
                (A, tA),
                (tA, tA),
                (tA, B),  # up B
                (0, B),
                (0, 0)    # close
            ]
            # If angle != 90°, rotate the B leg and the inside corner section
            if abs(angle_deg - 90.0) > 1e-6:
                theta = math.radians(90.0 - angle_deg)  # positive = open angle > 90 reduces rotation
                def rot(p):
                    x, y = p
                    xr = x*math.cos(theta) - y*math.sin(theta)
                    yr = x*math.sin(theta) + y*math.cos(theta)
                    return (xr, yr)
                pts = [rot(p) for p in pts]
            return pts

        l_poly_units = make_L_polygon(A, tA, B, tB, angle)

        # Quick preview
        prev = go.Figure()
        x = [p[0] for p in l_poly_units]; y = [p[1] for p in l_poly_units]
        prev.add_trace(go.Scatter(x=x+[x[0]], y=y+[y[0]], mode="lines+markers", name="L"))
        prev.update_yaxes(scaleanchor="x", scaleratio=1)
        prev.update_layout(title="L-shape preview", width=480, height=360, margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(prev, use_container_width=True)

    # ------- Add current shape -------
    with st.expander("Add the current shape to the Parts list", expanded=True):
        default_qty = st.number_input("Quantity", min_value=1, step=1, value=1)
        default_label = st.text_input("Label (optional)", value="")
        allow_rot = st.checkbox("Allow rotation for this part", value=True)
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
                        bw, bh = _bbox_of_polygon(pts)
                        w_u = round(bw/px_per_unit, precision); h_u = round(bh/px_per_unit, precision)
                        st.session_state.parts.append(Part(
                            id=str(uuid.uuid4()), label=default_label or f"Draw-{len(st.session_state.parts)+1}",
                            qty=int(default_qty), shape_type="polygon",
                            width=None, height=None, points=pts_units,
                            allow_rotation=bool(allow_rot)
                        ))
                        st.session_state.needs_nest = True
                        st.success(f"Added polygon (~{w_u} × {h_u} {_pretty_units(units)})")

            else:  # L-shape (parametric)
                if not l_poly_units or len(l_poly_units) < 3:
                    st.error("L-shape parameters invalid.")
                else:
                    st.session_state.parts.append(Part(
                        id=str(uuid.uuid4()), label=default_label or f"L-{len(st.session_state.parts)+1}",
                        qty=int(default_qty), shape_type="polygon",
                        width=None, height=None, points=l_poly_units,
                        allow_rotation=bool(allow_rot)
                    ))
                    st.session_state.needs_nest = True
                    st.success("Added L-shape polygon.")

    # ------- Parts editor -------
    st.markdown("---")
    st.subheader("Parts")
    if not st.session_state.parts:
        st.info("No parts yet. Add a shape above.")
    else:
        df = pd.DataFrame([{
            "id": p.id, "Label": p.label, "Type": p.shape_type,
            "Width": round(p.width, precision) if p.width is not None else None,
            "Height": round(p.height, precision) if p.height is not None else None,
            "Qty": p.qty, "Allow Rotation": p.allow_rotation
        } for p in st.session_state.parts])

        edited = st.data_editor(df.drop(columns=["id"]), use_container_width=True, num_rows="fixed", key="parts_editor")
        if len(edited) == len(st.session_state.parts):
            for i, p in enumerate(st.session_state.parts):
                row = edited.iloc[i]
                p.label = str(row["Label"]); p.allow_rotation = bool(row["Allow Rotation"]); p.qty = int(row["Qty"])
                if p.shape_type == "rect":
                    w = float(row["Width"]); h = float(row["Height"])
                    if (p.width != w) or (p.height != h):
                        p.width = max(0.0, w); p.height = max(0.0, h)
                        st.session_state.needs_nest = True

with right:
    st.subheader("Nesting")

    # ------- Auto-split logic for oversized rectangles -------
    def split_rect_if_needed(p: Part, sheet_w: float, sheet_h: float,
                             min_leg: float, seam_gap: float, prefer_long: bool) -> List[Part]:
        if p.shape_type != "rect" or p.width is None or p.height is None:
            return [p]
        W, H = p.width, p.height
        if W <= sheet_w and H <= sheet_h:
            return [p]

        # Decide split orientation
        long_side = "W" if W >= H else "H"
        split_along_width = (long_side == "W") if prefer_long else (long_side == "H")
        # But if only one dimension exceeds the sheet, force that dimension
        if W > sheet_w and H <= sheet_h: split_along_width = True
        if H > sheet_h and W <= sheet_w: split_along_width = False

        max_w = sheet_w; max_h = sheet_h
        parts_out: List[Part] = []

        if split_along_width:
            # Break W into N panels each <= max_w, insert seam gaps
            remaining = W
            idx = 1
            while remaining > 0:
                panel = min(max_w, remaining)
                if panel < min_leg:
                    # merge last two if possible
                    if parts_out:
                        prev = parts_out.pop()
                        panel += prev.width + seam_gap
                        new_w = panel
                        if new_w > max_w:  # still doesn't fit
                            st.session_state.messages.append(f"⚠️ Cannot split {p.label}: panel too short.")
                            return [p]  # fallback unsplit
                        parts_out.append(Part(
                            id=str(uuid.uuid4()), label=f"{p.label}-S{idx-1}",
                            qty=p.qty, shape_type="rect", width=new_w, height=H, points=None,
                            allow_rotation=p.allow_rotation
                        ))
                    else:
                        st.session_state.messages.append(f"⚠️ Cannot split {p.label}: min leg violated.")
                        return [p]
                else:
                    parts_out.append(Part(
                        id=str(uuid.uuid4()), label=f"{p.label}-S{idx}",
                        qty=p.qty, shape_type="rect", width=panel, height=H, points=None,
                        allow_rotation=p.allow_rotation
                    ))
                remaining -= panel
                if remaining > 0: remaining -= seam_gap  # account for seam kerf
                idx += 1
        else:
            # Split along height
            remaining = H
            idx = 1
            while remaining > 0:
                panel = min(max_h, remaining)
                if panel < min_leg:
                    if parts_out:
                        prev = parts_out.pop()
                        panel += prev.height + seam_gap
                        new_h = panel
                        if new_h > max_h:
                            st.session_state.messages.append(f"⚠️ Cannot split {p.label}: panel too short.")
                            return [p]
                        parts_out.append(Part(
                            id=str(uuid.uuid4()), label=f"{p.label}-S{idx-1}",
                            qty=p.qty, shape_type="rect", width=W, height=new_h, points=None,
                            allow_rotation=p.allow_rotation
                        ))
                    else:
                        st.session_state.messages.append(f"⚠️ Cannot split {p.label}: min leg violated.")
                        return [p]
                else:
                    parts_out.append(Part(
                        id=str(uuid.uuid4()), label=f"{p.label}-S{idx}",
                        qty=p.qty, shape_type="rect", width=W, height=panel, points=None,
                        allow_rotation=p.allow_rotation
                    ))
                remaining -= panel
                if remaining > 0: remaining -= seam_gap
                idx += 1

        return parts_out if parts_out else [p]

    # ------- Rectpack wrapper (polygons use bbox for now) -------
    def _rectpack(parts: List[Part], sheet_w: float, sheet_h: float, clearance: float, rotation: bool):
        packer = newPacker(rotation=rotation)
        to_add = []

        # Expand parts by qty; auto-split rectangles if enabled
        expanded: List[Part] = []
        for p in parts:
            if p.qty <= 0: continue
            subs = [p]
            if autosplit and p.shape_type == "rect":
                subs = split_rect_if_needed(p, sheet_w, sheet_h, min_leg, seam_gap, prefer_long_split == "long side")
            for s in subs:
                for _ in range(p.qty):
                    expanded.append(s)

        # Add to packer
        for s in expanded:
            if s.shape_type == "rect" and s.width and s.height:
                w = s.width + clearance; h = s.height + clearance
            elif s.shape_type == "polygon" and s.points:
                poly = Polygon(s.points); minx, miny, maxx, maxy = poly.bounds
                w = (maxx - minx) + clearance; h = (maxy - miny) + clearance
                if w > sheet_w + 1e-9 or h > sheet_h + 1e-9:
                    st.session_state.messages.append(f"⚠️ {s.label}: polygon (bbox {round(w-clearance,2)}×{round(h-clearance,2)}) exceeds sheet; cannot auto-split polygons.")
                    continue
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
        total_part_area = 0.0
        for abin in packer:
            rects = abin.rect_list()
            if not rects: continue
            placements = []
            for (x, y, w, h, rid) in rects:
                label = rid.split("#")[0]
                total_part_area += max(0.0, (w - clearance)) * max(0.0, (h - clearance))
                placements.append({"x": x, "y": y, "w": w - clearance, "h": h - clearance, "rid": rid, "label": label})
            sheets.append({"sheet_w": sheet_w, "sheet_h": sheet_h, "placements": placements})

        total_sheet_area = max(1e-9, len(sheets) * sheet_w * sheet_h)
        util = (total_part_area / total_sheet_area) if total_sheet_area else 0.0
        return sheets, util

    if st.button("🧩 Nest parts", type="primary", use_container_width=True):
        if not st.session_state.parts:
            st.warning("Add some parts first.")
        else:
            st.session_state.messages = []
            st.session_state.placements, st.session_state.utilization = _rectpack(
                st.session_state.parts, sheet_w, sheet_h, clearance, allow_rotation_global
            )
            st.session_state.needs_nest = False

    # ------- Results / Preview -------
    if st.session_state.messages:
        for m in st.session_state.messages:
            st.warning(m)

    if st.session_state.placements:
        util_pct = round(st.session_state.utilization * 100, 2)
        used_area = sum(p["w"] * p["h"] for s in st.session_state.placements for p in s["placements"])
        total_area = max(1e-9, len(st.session_state.placements) * sheet_w * sheet_h)

        st.markdown(f"**Sheets used:** {len(st.session_state.placements)}")
        st.markdown(f"**Utilization:** {util_pct}% (used {round(used_area,2)} / {round(total_area,2)} {_pretty_units(units)}²)")
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
