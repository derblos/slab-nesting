# app.py — Polygon-aware nesting + L-split policy + live drawer (postMessage) + bulk add + cutouts
from __future__ import annotations

import json, math, uuid
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
import plotly.graph_objects as go
from rectpack import newPacker
from shapely.affinity import rotate as shp_rotate, translate as shp_translate
from shapely.geometry import Polygon, box as shp_box

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import streamlit.components.v1 as components

# ---------------- Page setup ----------------
st.set_page_config(page_title="Nesting Tool", layout="wide")

# ---------------- Data model / state ----------------
@dataclass
class Part:
    id: str
    label: str
    qty: int
    shape_type: str            # "rect" or "polygon"
    width: Optional[float]
    height: Optional[float]
    points: Optional[List[Tuple[float, float]]]
    allow_rotation: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)

def _init_state():
    st.session_state.setdefault("parts", [])
    st.session_state.setdefault("needs_nest", True)
    st.session_state.setdefault("placements", [])
    st.session_state.setdefault("utilization", 0.0)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("draw_canvas_key", "draw_canvas")

_init_state()

def _pretty_units(u: str) -> str:
    return {"in": "in", "mm": "mm", "cm": "cm"}[u]

# ---------------- Fabric helpers ----------------
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

# ---------------- Cutout helper ----------------
def polygon_with_cutouts(outer_pts: List[Tuple[float,float]], cutouts: List[Tuple[float,float,float,float]]) -> Polygon:
    poly = Polygon(outer_pts)
    for (cx, cy, cw, ch) in cutouts:
        poly = poly.difference(shp_box(cx, cy, cx+cw, cy+ch))
    return poly

# ---------------- Konva live rectangle drawer (with postMessage Add) ----------------
def render_konva_rect_drawer(units_label="in", ppu=8.0, width_px=1000, height_px=600):
    payload = json.dumps({"units": units_label, "ppu": ppu, "w": int(width_px), "h": int(height_px)})
    html = f"""
    <style>
      #konva-wrap {{ position: relative; width: {int(width_px)}px; }}
      #konva-live-root {{ }}
      #add-rect-btn {{
        position: absolute; right: 10px; bottom: 10px;
        padding: 8px 12px; border: 1px solid #aaa; border-radius: 6px;
        background: #fff; cursor: pointer; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      }}
    </style>
    <div id="konva-wrap">
      <div id="konva-live-root"></div>
      <button id="add-rect-btn">➕ Add this rectangle</button>
    </div>

    <script src="https://unpkg.com/konva@9.3.3/konva.min.js"></script>
    <script>
      const cfg = {payload};
      const scale = cfg.ppu;

      const container = document.createElement('div');
      container.style.border = '1px solid #ddd';
      container.style.width = cfg.w + 'px';
      container.style.height = cfg.h + 'px';
      document.getElementById('konva-live-root').appendChild(container);

      const stage = new Konva.Stage({{ container: container, width: cfg.w, height: cfg.h }});
      const layer = new Konva.Layer(); stage.add(layer);

      let start = null, rect = null, label = null;

      const hint = new Konva.Text({{
        x: 8, y: 8, text: 'Drag to draw a rectangle', fontSize: 14, fill: '#333'
      }});
      layer.add(hint);

      function fmt(v) {{ return (v/scale).toFixed(2); }}

      stage.on('mousedown', () => {{
        const pos = stage.getPointerPosition();
        start = {{ x: pos.x, y: pos.y }};
        if (rect) rect.destroy();
        if (label) label.destroy();
        rect = new Konva.Rect({{
          x: start.x, y: start.y, width: 0, height: 0,
          stroke: '#4074f4', strokeWidth: 1, dash: [4, 3]
        }});
        label = new Konva.Label({{ x: start.x + 8, y: start.y - 26 }});
        label.add(new Konva.Tag({{ fill: 'rgba(255,255,255,0.9)', stroke: '#aaa' }}));
        label.add(new Konva.Text({{
          text: '', fontSize: 12, fill: '#111', padding: 4
        }}));
        layer.add(rect); layer.add(label);
        layer.draw();
      }});

      stage.on('mousemove', () => {{
        if (!start || !rect) return;
        const pos = stage.getPointerPosition();
        const w = pos.x - start.x;
               const h = pos.y - start.y;
        rect.width(w); rect.height(h);
        const txt = label.getChildren()[1];
        txt.text(fmt(Math.abs(w)) + ' × ' + fmt(Math.abs(h)) + ' {units_label}');
        label.position({{ x: start.x + 8, y: start.y - 26 }});
        layer.batchDraw();
      }});

      stage.on('mouseup', () => {{ start = null; }});

      function getRect() {{
        if (!rect) return {{ w: 0, h: 0 }};
        return {{ w: Math.abs(rect.width())/scale, h: Math.abs(rect.height())/scale }};
      }}

      // Send dims to parent via postMessage; Streamlit listens (we add a small listener below in Python)
      document.getElementById('add-rect-btn').addEventListener('click', () => {{
        const data = getRect();
        window.parent.postMessage({{ type: "live_rect", data }}, "*");
      }});
    </script>
    """
    components.html(html, height=int(height_px)+60, scrolling=False)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Project settings")
    units = st.selectbox("Units", ["in", "mm", "cm"], index=0, key="sb_units")
    precision = st.number_input("Precision (decimals)", 0, 4, 2, key="sb_precision")
    px_per_unit = st.slider("Canvas scale (px per unit)", 2, 20, 8, key="sb_ppu")

    st.markdown("### Drawing canvas size")
    canvas_w = st.number_input("Canvas width (px)", min_value=600, max_value=2400, value=1200, step=50, key="sb_canvas_w")
    canvas_h = st.number_input("Canvas height (px)", min_value=400, max_value=2000, value=850, step=50, key="sb_canvas_h")

    st.markdown("### Sheet")
    sheet_w = st.number_input(f"Sheet width ({_pretty_units(units)})", min_value=1.0, value=97.0, step=1.0, format="%.2f", key="sb_sheet_w")
    sheet_h = st.number_input(f"Sheet height ({_pretty_units(units)})", min_value=1.0, value=80.50, step=1.0, format="%.2f", key="sb_sheet_h")
    clearance = st.number_input(f"Clearance between parts ({_pretty_units(units)})", min_value=0.0, value=0.25, step=0.05, format="%.2f", key="sb_clearance")
    allow_rotation_global = st.checkbox("Allow rotation globally (0/90°)", value=True, key="sb_allow_rot")

    st.markdown("### Nesting mode")
    mode = st.radio("Mode", ["Fast (rectpack)", "Precision (polygon-aware)"], index=1, key="sb_mode")

    st.markdown("### Auto-split oversized rectangles")
    autosplit_rects = st.checkbox("Auto-split rectangles that exceed sheet size", value=True, key="sb_autosplit")
    seam_gap = st.number_input(f"Seam/kerf gap at split ({_pretty_units(units)})", min_value=0.0, value=0.125, step=0.01, format="%.3f", key="sb_seam_gap")
    min_leg = st.number_input(f"Minimum leg length after split ({_pretty_units(units)})", min_value=1.0, value=6.0, step=0.5, format="%.2f", key="sb_min_leg")
    prefer_long_split = st.selectbox("Prefer split along", ["long side", "short side"], index=0, key="sb_prefer_split")

    st.markdown("### L-shape split policy")
    enable_L_seams = st.checkbox("Allow seams on L-shapes when needed", value=True, key="sb_L_seams")
    L_max_leg_no_split = st.number_input(f"Max leg length without split ({_pretty_units(units)})", min_value=1.0, value=48.0, step=1.0, format="%.0f", key="sb_L_max_leg")

    st.markdown("---")
    if st.button("🗑️ Clear project", use_container_width=True, type="primary", key="sb_clear_project"):
        for k in ["parts", "needs_nest", "placements", "utilization", "messages", "draw_canvas_key"]:
            st.session_state.pop(k, None)
        _init_state()
        st.success("Project cleared.")

# ---------------- Utility: query param helpers + postMessage listener ----------------
def _get_query_params():
    if hasattr(st, "query_params"):
        return st.query_params
    try:
        return st.experimental_get_query_params()
    except Exception:
        return {}

def _clear_query_params():
    try:
        if hasattr(st, "query_params"):
            st.query_params.clear()
        else:
            st.experimental_set_query_params()
    except Exception:
        pass

# Listen for postMessage from the Konva iframe; when received, write ?live_rect=... to parent URL
components.html("""
<script>
window.addEventListener("message", (event) => {
  if (event && event.data && event.data.type === "live_rect") {
    try {
      const vals = JSON.stringify(event.data.data || {w:0,h:0});
      const url = new URL(window.location.href);
      url.searchParams.set("live_rect", vals);
      window.location.href = url.toString(); // triggers Streamlit rerun
    } catch (e) { /* no-op */ }
  }
});
</script>
""", height=0)

# ---------------- Main layout (single column; nesting moved below) ----------------
st.title("Nesting Tool")

# Tool chooser
tool = st.radio(
    "Tool",
    ["Rectangle", "Polygon (freehand)", "L-shape (parametric)"],
    horizontal=True,
    key="tool_radio"
)

# Toggle live drawer for rectangles
use_live_drawer = False
if tool == "Rectangle":
    use_live_drawer = st.toggle(
        "Use live drawer (beta)",
        value=True,
        help="Shows W×H while dragging (Konva). Turn off to use classic canvas with bulk-add.",
        key="toggle_live_drawer"
    )

last_obj = None
live_w = live_h = None
canvas_result = None

# ---------- Live drawer path ----------
if tool == "Rectangle" and use_live_drawer:
    render_konva_rect_drawer(
        units_label=_pretty_units(units),
        ppu=float(px_per_unit),
        width_px=int(canvas_w),
        height_px=int(canvas_h)
    )

    lc1, lc2, lc3 = st.columns([2,1,1])
    with lc1:
        live_label = st.text_input("Label (optional)", value="", key="live_rect_label")
    with lc2:
        live_qty = st.number_input("Qty", min_value=1, value=1, step=1, key="live_rect_qty")
    with lc3:
        live_allow_rot = st.checkbox("Allow rotation", value=True, key="live_rect_rot")

    st.markdown("**Optional: rectangular cutouts for this rectangle**")
    add_cut_live = st.checkbox("Add cutout(s) to this rectangle", key="live_rect_cut_toggle")
    cut_list_live: List[Tuple[float,float,float,float]] = []
    if add_cut_live:
        ncuts_live = st.number_input("Number of cutouts", min_value=1, max_value=5, value=1, step=1, key="live_rect_cut_count")
        for i in range(ncuts_live):
            st.write(f"Cutout #{i+1}")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                cw = st.number_input(f"W{i+1} ({_pretty_units(units)})", min_value=0.1, value=30.0, step=0.1, format="%.2f", key=f"lr_cw_{i}")
            with c2:
                ch = st.number_input(f"H{i+1} ({_pretty_units(units)})", min_value=0.1, value=20.0, step=0.1, format="%.2f", key=f"lr_ch_{i}")
            with c3:
                cx = st.number_input(f"X{i+1} offset", value=10.0, step=0.5, format="%.2f", key=f"lr_cx_{i}")
            with c4:
                cy = st.number_input(f"Y{i+1} offset", value=10.0, step=0.5, format="%.2f", key=f"lr_cy_{i}")
            cut_list_live.append((cx, cy, cw, ch))

# ---------- Classic canvas path (rect/polygon drawing + bulk add) ----------
if (tool != "Rectangle") or (tool == "Rectangle" and not use_live_drawer):
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

    col_live = st.columns(2)
    with col_live[0]:
        st.metric("Live width", f"{live_w if live_w is not None else '—'} {_pretty_units(units)}")
    with col_live[1]:
        st.metric("Live height", f"{live_h if live_h is not None else '—'} {_pretty_units(units)}")

    bulk_cols = st.columns([1,1,1])
    with bulk_cols[0]:
        if st.button("➕ Add all drawn shapes", type="secondary", key="btn_add_all_drawn"):
            added = 0
            if canvas_result and canvas_result.json_data:
                objs = canvas_result.json_data.get("objects", [])
                for obj in objs:
                    if obj.get("type") == "rect":
                        w_px, h_px = _fabric_rect_dims(obj)
                        w = round(w_px / px_per_unit, precision)
                        h = round(h_px / px_per_unit, precision)
                        if w > 0 and h > 0:
                            st.session_state.parts.append(Part(
                                id=str(uuid.uuid4()), label=f"Draw-{len(st.session_state.parts)+1}",
                                qty=1, shape_type="rect", width=float(w), height=float(h),
                                points=None, allow_rotation=True
                            ))
                            added += 1
                    elif obj.get("path"):
                        pts = _fabric_polygon_points(obj)
                        if len(pts) >= 3:
                            pts_units = [(x/px_per_unit, y/px_per_unit) for (x,y) in pts]
                            st.session_state.parts.append(Part(
                                id=str(uuid.uuid4()), label=f"Draw-{len(st.session_state.parts)+1}",
                                qty=1, shape_type="polygon", width=None, height=None,
                                points=pts_units, allow_rotation=True
                            ))
                            added += 1
            if added:
                st.session_state.needs_nest = True
                st.success(f"Added {added} shape(s) from the drawing layer.")
    with bulk_cols[1]:
        if st.button("🧹 Clear drawing layer", key="btn_clear_layer"):
            st.session_state["draw_canvas_key"] = uuid.uuid4().hex
            st.rerun()
    with bulk_cols[2]:
        st.caption("Tip: draw multiple shapes first, then **Add all**.")

# ---------- Parametric L-shape ----------
l_poly_units = None; A=B=tA=tB=angle=None
if tool == "L-shape (parametric)":
    st.write("Define outer legs A & B, depth D (tA=tB), and angle (defaults 90°).")
    colA, colB = st.columns(2)
    with colA:
        A = st.number_input(f"Outer leg A length ({_pretty_units(units)})", min_value=1.0, value=120.0, step=0.5, format="%.2f", key="L_A")
        tA = st.number_input(f"Depth / thickness D ({_pretty_units(units)})", min_value=0.1, value=25.0, step=0.1, format="%.2f", key="L_D")
    with colB:
        B = st.number_input(f"Outer leg B length ({_pretty_units(units)})", min_value=1.0, value=60.0, step=0.5, format="%.2f", key="L_B")
        tB = tA
    angle = st.number_input("Inside angle (degrees)", min_value=30.0, max_value=150.0, value=90.0, step=1.0, format="%.0f", key="L_angle")

    def make_L_polygon(A, D, B, angle_deg) -> List[Tuple[float, float]]:
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
    prev.update_layout(title="L preview", width=520, height=380, margin=dict(l=20,r=20,t=40,b=20))
    st.plotly_chart(prev, use_container_width=True)

# ---------- Classic expander to add a single current shape (for classic tools + L) ----------
with st.expander("Add the current shape to the Parts list (classic tools)", expanded=(tool!="Rectangle" or not use_live_drawer)):
    default_qty = st.number_input("Quantity", min_value=1, step=1, value=1, key="exp_qty")
    default_label = st.text_input("Label (optional)", value="", key="exp_label")
    allow_rot = st.checkbox("Allow rotation for this part (0/90°)", value=True, key="exp_rot")

    st.markdown("**Optional: rectangular cutouts (sinks, cooktops)**")
    add_cut = st.checkbox("Add cutout(s) to this part", key="exp_cut_toggle")
    cut_list: List[Tuple[float,float,float,float]] = []
    if add_cut:
        ncuts = st.number_input("Number of cutouts", min_value=1, max_value=5, value=1, step=1, key="exp_cut_count")
        for i in range(ncuts):
            st.write(f"Cutout #{i+1}")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                cw = st.number_input(f"W{i+1} ({_pretty_units(units)})", min_value=0.1, value=30.0, step=0.1, format="%.2f", key=f"exp_cw_{i}")
            with c2:
                ch = st.number_input(f"H{i+1} ({_pretty_units(units)})", min_value=0.1, value=20.0, step=0.1, format="%.2f", key=f"exp_ch_{i}")
            with c3:
                cx = st.number_input(f"X{i+1} offset", value=10.0, step=0.5, format="%.2f", key=f"exp_cx_{i}")
            with c4:
                cy = st.number_input(f"Y{i+1} offset", value=10.0, step=0.5, format="%.2f", key=f"exp_cy_{i}")
            cut_list.append((cx, cy, cw, ch))

    if st.button("➕ Add this shape", type="secondary", key="exp_add_btn"):
        if tool == "Rectangle" and not use_live_drawer:
            if not last_obj or last_obj.get("type") != "rect":
                st.warning("Draw a rectangle first.")
            else:
                w_px, h_px = _fabric_rect_dims(last_obj)
                w = round(w_px / px_per_unit, precision); h = round(h_px / px_per_unit, precision)
                if w <= 0 or h <= 0:
                    st.error("This rectangle has zero width/height.")
                else:
                    if add_cut and cut_list:
                        outer = [(0,0),(w,0),(w,h),(0,h),(0,0)]
                        poly = polygon_with_cutouts(outer, cut_list)
                        st.session_state.parts.append(Part(
                            id=str(uuid.uuid4()), label=default_label or f"Rect-{len(st.session_state.parts)+1}",
                            qty=int(default_qty), shape_type="polygon",
                            width=None, height=None, points=list(poly.exterior.coords),
                            allow_rotation=bool(allow_rot),
                            meta={"cutouts": cut_list}
                        ))
                        st.success(f"Added rectangle (with {len(cut_list)} cutout(s))")
                    else:
                        st.session_state.parts.append(Part(
                            id=str(uuid.uuid4()), label=default_label or f"Rect-{len(st.session_state.parts)+1}",
                            qty=int(default_qty), shape_type="rect",
                            width=float(w), height=float(h), points=None,
                            allow_rotation=bool(allow_rot)
                        ))
                        st.success(f"Added rectangle ({w} × {h} {_pretty_units(units)})")
                    st.session_state.needs_nest = True

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
                        id=str(uuid.uuid4()), label=default_label or f"Poly-{len(st.session_state.parts)+1}",
                        qty=int(default_qty), shape_type="polygon",
                        width=None, height=None, points=pts_units,
                        allow_rotation=bool(allow_rot)
                    ))
                    st.session_state.needs_nest = True
                    st.success("Added polygon.")

        elif tool == "L-shape (parametric)":
            if not l_poly_units or len(l_poly_units) < 3:
                st.error("L-shape parameters invalid.")
            else:
                meta = {"is_L": True, "A": float(A), "B": float(B), "tA": float(tA), "tB": float(tB), "angle": float(angle)}
                if add_cut and cut_list:
                    poly = polygon_with_cutouts(l_poly_units, cut_list)
                    meta["cutouts"] = cut_list
                    pts_out = list(poly.exterior.coords)
                else:
                    pts_out = l_poly_units
                st.session_state.parts.append(Part(
                    id=str(uuid.uuid4()), label=default_label or f"L-{len(st.session_state.parts)+1}",
                    qty=int(default_qty), shape_type="polygon",
                    width=None, height=None, points=pts_out,
                    allow_rotation=bool(allow_rot),
                    meta=meta
                ))
                st.session_state.needs_nest = True
                st.success("Added L-shape" + (" (with cutout(s))." if (add_cut and cut_list) else "."))

# ---------- Handle live_rect query param (set by postMessage listener) ----------
qp = _get_query_params()
raw = qp.get("live_rect")
if isinstance(raw, list):
    raw = raw[0] if raw else None
if raw:
    try:
        vals = json.loads(raw)
        w = float(vals.get("w", 0.0))
        h = float(vals.get("h", 0.0))
        if w <= 0 or h <= 0:
            st.warning("Draw a rectangle first (drag on the canvas).")
        else:
            if ('live_rect_cut_toggle' in st.session_state and st.session_state['live_rect_cut_toggle']) and ('lr_cw_0' in st.session_state):
                # Rebuild cutouts from state (if any)
                cut_list_live_re = []
                n_guess = 10
                for i in range(n_guess):
                    cw_k, ch_k, cx_k, cy_k = f"lr_cw_{i}", f"lr_ch_{i}", f"lr_cx_{i}", f"lr_cy_{i}"
                    if cw_k in st.session_state and ch_k in st.session_state and cx_k in st.session_state and cy_k in st.session_state:
                        cut_list_live_re.append((
                            float(st.session_state[cx_k]),
                            float(st.session_state[cy_k]),
                            float(st.session_state[cw_k]),
                            float(st.session_state[cy_k] if False else st.session_state[ch_k])  # just to be explicit; same as ch_k
                        ))
                if cut_list_live_re:
                    outer = [(0,0),(w,0),(w,h),(0,h),(0,0)]
                    poly = polygon_with_cutouts(outer, cut_list_live_re)
                    st.session_state.parts.append(Part(
                        id=str(uuid.uuid4()),
                        label=(st.session_state.get("live_rect_label") or f"Rect-{len(st.session_state.parts)+1}"),
                        qty=int(st.session_state.get("live_rect_qty", 1)),
                        shape_type="polygon",
                        width=None, height=None,
                        points=list(poly.exterior.coords),
                        allow_rotation=bool(st.session_state.get("live_rect_rot", True)),
                        meta={"cutouts": cut_list_live_re}
                    ))
                    st.success(f"Added rectangle with {len(cut_list_live_re)} cutout(s)")
                else:
                    st.session_state.parts.append(Part(
                        id=str(uuid.uuid4()),
                        label=(st.session_state.get("live_rect_label") or f"Rect-{len(st.session_state.parts)+1}"),
                        qty=int(st.session_state.get("live_rect_qty", 1)),
                        shape_type="rect",
                        width=w, height=h, points=None,
                        allow_rotation=bool(st.session_state.get("live_rect_rot", True))
                    ))
                    st.success(f"Added rectangle ({round(w,st.session_state.get('sb_precision',2))} × {round(h,st.session_state.get('sb_precision',2))} {_pretty_units(st.session_state.get('sb_units','in'))})")
            else:
                st.session_state.parts.append(Part(
                    id=str(uuid.uuid4()),
                    label=(st.session_state.get("live_rect_label") or f"Rect-{len(st.session_state.parts)+1}"),
                    qty=int(st.session_state.get("live_rect_qty", 1)),
                    shape_type="rect",
                    width=w, height=h, points=None,
                    allow_rotation=bool(st.session_state.get("live_rect_rot", True))
                ))
                st.success(f"Added rectangle ({round(w,st.session_state.get('sb_precision',2))} × {round(h,st.session_state.get('sb_precision',2))} {_pretty_units(st.session_state.get('sb_units','in'))})")
            st.session_state.needs_nest = True
    except Exception as e:
        st.warning(f"Could not read live rectangle: {e}")
    finally:
        _clear_query_params()

# ---------- Parts editor ----------
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

# ===================== Nesting (moved BELOW drawing & parts) =====================
st.markdown("---")
st.header("Nesting")

# -------- Split helpers --------
def split_rect_if_needed(p: Part, sheet_w: float, sheet_h: float,
                         min_leg: float, seam_gap: float, prefer_long: bool, clearance: float) -> tuple[list[Part], bool]:
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
    if W > usable_w and H <= usable_h:
        split_along_width = True
    if H > usable_h and W <= usable_w:
        split_along_width = False

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
            if remaining > 1e-9:
                remaining -= seam_gap
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
            if remaining > 1e-9:
                remaining -= seam_gap
            idx += 1

    return (parts_out if parts_out else [p]), (len(parts_out) > 1)

def decompose_L_prefer_corner(p: Part) -> list[Part]:
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

# -------- Nesting engines --------
def _rectpack_trial(parts_trial, sheet_w, sheet_h, clearance, rotation):
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

def rectpack_nest(parts: List[Part], sheet_w: float, sheet_h: float, clearance: float, rotation: bool):
    to_add = []
    expanded: List[Part] = []
    any_split = False

    for p in parts:
        if p.qty <= 0: continue
        subs = [p]
        did_split = False

        if p.meta.get("is_L") and abs(p.meta.get("angle", 90.0) - 90.0) < 1e-6:
            cand1 = decompose_L_prefer_corner(p)
            cand2 = decompose_L_alternate(p)
            s1, _ = _rectpack_trial(cand1, sheet_w, sheet_h, clearance, rotation)
            s2, _ = _rectpack_trial(cand2, sheet_w, sheet_h, clearance, rotation)
            subs = cand1 if len(s1) <= len(s2) else cand2
            did_split = True

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

    for (w, h, rid, _) in to_add:
        packer.add_rect(w, h, rid=rid)
    for _ in range(200):
        packer.add_bin(sheet_w, sheet_h)
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
            placements.append({"x": x, "y": y, "w": w - clearance, "h": h - clearance, "rid": rid, "label": label})
            total_area += max(0.0, (w - clearance)) * max(0.0, (h - clearance))
        sheets.append({"sheet_w": sheet_w, "sheet_h": sheet_h, "placements": placements})

    util = total_area / max(1e-9, len(sheets)*sheet_w*sheet_h)
    return sheets, util

def _poly_to_rect_anno(poly: Polygon, label: str, precision: int):
    minx, miny, maxx, maxy = poly.bounds
    w = maxx - minx; h = maxy - miny
    return {"x": float(minx), "y": float(miny), "w": float(w), "h": float(h), "rid": f"{label}#{uuid.uuid4().hex[:4]}", "label": label,
            "exact_w": round(w, precision), "exact_h": round(h, precision)}

def poly_nest(parts: List[Part], sheet_w: float, sheet_h: float, clearance: float, rotation: bool, grid_step: float = 0.5):
    expanded: List[Tuple[Part, Polygon]] = []
    any_split = False

    def part_to_poly(s: Part) -> Optional[Polygon]:
        if s.shape_type == "rect" and s.width and s.height:
            outer = shp_box(0, 0, s.width, s.height)
            if s.meta.get("cutouts"):
                diff = outer
                for (cx,cy,cw,ch) in s.meta["cutouts"]:
                    diff = diff.difference(shp_box(cx, cy, cx+cw, cy+ch))
                return diff
            return outer
        if s.shape_type == "polygon" and s.points:
            try:
                poly = Polygon(s.points)
                if s.meta.get("cutouts"):
                    for (cx,cy,cw,ch) in s.meta["cutouts"]:
                        poly = poly.difference(shp_box(cx, cy, cx+cw, cy+ch))
                return poly
            except Exception:
                return None
        return None

    for p in parts:
        if p.qty <= 0: continue
        subs = [p]; did_split = False

        if p.meta.get("is_L") and abs(p.meta.get("angle",90.0)-90.0) < 1e-6:
            cand1 = decompose_L_prefer_corner(p)
            cand2 = decompose_L_alternate(p)
            s1, _ = _rectpack_trial(cand1, sheet_w, sheet_h, clearance, rotation)
            s2, _ = _rectpack_trial(cand2, sheet_w, sheet_h, clearance, rotation)
            subs = cand1 if len(s1) <= len(s2) else cand2
            did_split = True

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

    placed = []
    sheet_poly = shp_box(0, 0, sheet_w, sheet_h)
    buffer_clear = clearance / 2.0 if clearance > 0 else 0.0

    for s, base_poly in sorted(expanded, key=lambda t: t[1].area, reverse=True):
        orientations = [0] + ([90] if rotation_effective else [])
        placed_ok = False

        for ang in orientations:
            poly = shp_rotate(base_poly, ang, origin=(0,0), use_radians=False)
            y = 0.0
            while y <= sheet_h and not placed_ok:
                x = 0.0
                while x <= sheet_w and not placed_ok:
                    test = shp_translate(poly, xoff=x, yoff=y)
                    bounds_ok = test.buffer(buffer_clear).within(sheet_poly)
                    if not bounds_ok:
                        x += grid_step; continue
                    collision = any(test.buffer(buffer_clear).intersects(o["poly"].buffer(buffer_clear)) for o in placed)
                    if not collision:
                        placed.append({"poly": test, "label": s.label, "rid": f"{s.label}#{uuid.uuid4().hex[:4]}", "angle": ang})
                        placed_ok = True
                        break
                    x += grid_step
                y += grid_step
            if placed_ok: break

        if not placed_ok:
            yield {"sheet_w": sheet_w, "sheet_h": sheet_h, "placements": [
                _poly_to_rect_anno(d["poly"], d["label"], st.session_state.get("sb_precision", 2)) for d in placed
            ]}
            placed = []
            # try origin on new sheet
            for ang in ([0, 90] if rotation_effective else [0]):
                poly_try = shp_rotate(base_poly, ang, origin=(0,0), use_radians=False)
                if poly_try.buffer(buffer_clear).within(sheet_poly):
                    placed.append({"poly": poly_try, "label": s.label, "rid": f"{s.label}#{uuid.uuid4().hex[:4]}", "angle": ang})
                    break
            if not placed:
                st.session_state.messages.append(f"⚠️ {s.label}: cannot place on empty sheet (too large).")
    if placed:
        yield {"sheet_w": sheet_w, "sheet_h": sheet_h, "placements": [
            _poly_to_rect_anno(d["poly"], d["label"], st.session_state.get("sb_precision", 2)) for d in placed
        ]}

# -------- Run nesting --------
do_nest = st.button("🧩 Nest parts", type="primary", use_container_width=True, key="btn_nest")
if do_nest:
    if not st.session_state.parts:
        st.warning("Add some parts first.")
    else:
        st.session_state.messages = []
        if mode == "Fast (rectpack)":
            sheets, util = rectpack_nest(st.session_state.parts, sheet_w, sheet_h, clearance, allow_rotation_global)
            st.session_state.placements, st.session_state.utilization = sheets, util
        else:
            sheets = list(poly_nest(st.session_state.parts, sheet_w, sheet_h, clearance, allow_rotation_global, grid_step=0.5))
            placed_area = sum(p["w"] * p["h"] for s in sheets for p in s["placements"])
            util = placed_area / max(1e-9, len(sheets)*sheet_w*sheet_h)  # lower bound (bbox)
            st.session_state.placements, st.session_state.utilization = sheets, util
        st.session_state.needs_nest = False

# Messages
if st.session_state.messages:
    for m in st.session_state.messages:
        st.warning(m)

# -------- Preview --------
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
                               text=f"{label}<br>{round(w, st.session_state.get('sb_precision',2))} × {round(h, st.session_state.get('sb_precision',2))}",
                               showarrow=False, font=dict(size=12))
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(
            title=f"Sheet {si}", width=1000, height=700, margin=dict(l=30, r=30, t=50, b=30),
            xaxis=dict(range=[-1, W+1], title=f"Width ({_pretty_units(units)})"),
            yaxis=dict(range=[H+1, -1], title=f"Height ({_pretty_units(units)})"),
            dragmode="pan",
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No nesting results yet. Click **Nest parts** when you’re ready.")
