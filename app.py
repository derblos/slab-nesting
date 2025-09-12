# app.py  (fixed: removed unsupported 'realtime_update' arg)
from __future__ import annotations

import uuid
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

# ────────────── Models ──────────────
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

    @property
    def area(self) -> float:
        if self.shape_type == "rect" and self.width and self.height:
            return self.width * self.height
        if self.shape_type == "polygon" and self.points:
            try:
                return Polygon(self.points).area
            except Exception:
                return 0.0
        return 0.0

def _init_state():
    st.session_state.setdefault("parts", [])
    st.session_state.setdefault("needs_nest", True)
    st.session_state.setdefault("placements", [])
    st.session_state.setdefault("utilization", 0.0)

_init_state()

# ────────────── Helpers ──────────────
def _fabric_rect_dims(obj: Dict) -> Tuple[float, float]:
    w = float(obj.get("width", 0)) * float(obj.get("scaleX", 1.0))
    h = float(obj.get("height", 0)) * float(obj.get("scaleY", 1.0))
    return abs(w), abs(h)

def _fabric_polygon_points(obj: Dict) -> List[Tuple[float, float]]:
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
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (max(xs) - min(xs), max(ys) - min(ys))

def _pretty_units(u: str) -> str:
    return {"in": "in", "mm": "mm", "cm": "cm"}[u]

# ────────────── Sidebar ──────────────
with st.sidebar:
    st.header("Project settings")
    units = st.selectbox("Units", ["in", "mm", "cm"], index=0)
    precision = st.number_input("Precision (decimal places)", 0, 4, 2)
    px_per_unit = st.slider("Canvas scale (px per unit)", 2, 20, 8)

    st.markdown("---")
    st.subheader("Sheet")
    sheet_w = st.number_input(f"Sheet width ({_pretty_units(units)})", min_value=1.0, value=97.0, step=1.0, format="%.2f")
    sheet_h = st.number_input(f"Sheet height ({_pretty_units(units)})", min_value=1.0, value=80.50, step=1.0, format="%.2f")
    clearance = st.number_input(f"Clearance between parts ({_pretty_units(units)})", min_value=0.0, value=0.25, step=0.05, format="%.2f")
    allow_rotation_global = st.checkbox("Allow rotation globally", value=True)

    st.markdown("---")
    if st.button("🗑️ Clear project", use_container_width=True, type="primary"):
        for k in ["parts", "needs_nest", "placements", "utilization"]:
            st.session_state.pop(k, None)
        _init_state()
        st.success("Project cleared.")

# ────────────── Main ──────────────
left, right = st.columns([0.55, 0.45])

with left:
    st.title("Nesting Tool — Draw Only")
    st.subheader("Draw a part")

    tool = st.radio("Tool", ["Rectangle", "Polygon"], horizontal=True)
    drawing_mode = "rect" if tool == "Rectangle" else "polygon"

    canvas_w, canvas_h = 900, 520
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0.0)",
        stroke_width=2,
        stroke_color="#1f77b4",
        background_color="#fafafa",
        height=canvas_h,
        width=canvas_w,
        drawing_mode=drawing_mode,
        display_toolbar=True,
        update_streamlit=True,   # keep this; remove 'realtime_update'
        key="draw_canvas"
    )

    # Live dimensions (without 'realtime_update')
    live_w = live_h = None
    last_obj = None
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

    st.caption(
        f"**Live dimensions**: "
        + (f"{live_w} × {live_h} {_pretty_units(units)}" if live_w is not None else "— start/continue drawing —")
    )

    with st.expander("Add the current shape to the Parts list", expanded=True):
        default_qty = st.number_input("Quantity", min_value=1, step=1, value=1)
        default_label = st.text_input("Label (optional)", value="")
        allow_rot = st.checkbox("Allow rotation for this part", value=True)
        add_btn = st.button("➕ Add this shape", type="secondary")

        if add_btn:
            if not last_obj:
                st.warning("Draw a shape first, then click add.")
            else:
                if drawing_mode == "rect" and last_obj.get("type") == "rect":
                    w_px, h_px = _fabric_rect_dims(last_obj)
                    w = round(w_px / px_per_unit, precision)
                    h = round(h_px / px_per_unit, precision)
                    if w <= 0 or h <= 0:
                        st.error("This rectangle has zero width/height.")
                    else:
                        st.session_state.parts.append(Part(
                            id=str(uuid.uuid4()),
                            label=default_label or f"Draw-{len(st.session_state.parts)+1}",
                            qty=int(default_qty),
                            shape_type="rect",
                            width=float(w), height=float(h),
                            points=None,
                            allow_rotation=bool(allow_rot),
                        ))
                        st.session_state.needs_nest = True
                        st.success(f"Added rectangle ({w} × {h} {_pretty_units(units)})")
                else:
                    pts = _fabric_polygon_points(last_obj)
                    if len(pts) < 3:
                        st.error("Polygon needs at least 3 points.")
                    else:
                        pts_units = [(x / px_per_unit, y / px_per_unit) for (x, y) in pts]
                        bw, bh = _bbox_of_polygon(pts)
                        w_u = round(bw / px_per_unit, precision)
                        h_u = round(bh / px_per_unit, precision)
                        st.session_state.parts.append(Part(
                            id=str(uuid.uuid4()),
                            label=default_label or f"Draw-{len(st.session_state.parts)+1}",
                            qty=int(default_qty),
                            shape_type="polygon",
                            width=None, height=None,
                            points=pts_units,
                            allow_rotation=bool(allow_rot),
                        ))
                        st.session_state.needs_nest = True
                        st.success(f"Added polygon (~{w_u} × {h_u} {_pretty_units(units)})")

    st.markdown("---")
    st.subheader("Parts")
    if not st.session_state.parts:
        st.info("No parts yet. Draw a shape and click **Add this shape**.")
    else:
        df = pd.DataFrame([{
            "id": p.id,
            "Label": p.label,
            "Type": p.shape_type,
            "Width": round(p.width, precision) if p.width is not None else None,
            "Height": round(p.height, precision) if p.height is not None else None,
            "Qty": p.qty,
            "Allow Rotation": p.allow_rotation
        } for p in st.session_state.parts])

        edited = st.data_editor(
            df.drop(columns=["id"]),
            use_container_width=True,
            num_rows="fixed",
            key="parts_editor",
        )

        if len(edited) == len(st.session_state.parts):
            for i, p in enumerate(st.session_state.parts):
                row = edited.iloc[i]
                p.label = str(row["Label"])
                p.allow_rotation = bool(row["Allow Rotation"])
                p.qty = int(row["Qty"])
                if p.shape_type == "rect":
                    w = float(row["Width"])
                    h = float(row["Height"])
                    if (p.width != w) or (p.height != h):
                        p.width = max(0.0, w)
                        p.height = max(0.0, h)
                        st.session_state.needs_nest = True

with right:
    st.subheader("Nesting")

    def _rectpack(parts: List[Part], sheet_w: float, sheet_h: float, clearance: float, rotation: bool):
        packer = newPacker(rotation=rotation)
        to_add = []
        for p in parts:
            if p.qty <= 0:
                continue
            if p.shape_type == "rect" and p.width and p.height:
                w = p.width + clearance
                h = p.height + clearance
            elif p.shape_type == "polygon" and p.points:
                poly = Polygon(p.points)
                minx, miny, maxx, maxy = poly.bounds
                w = (maxx - minx) + clearance
                h = (maxy - miny) + clearance
            else:
                continue
            for q in range(p.qty):
                rid = f"{p.label}#{q+1}"
                to_add.append((w, h, rid, p.label))

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
            st.session_state.placements, st.session_state.utilization = _rectpack(
                st.session_state.parts, sheet_w, sheet_h, clearance, allow_rotation_global
            )
            st.session_state.needs_nest = False

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
