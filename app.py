# app.py
# Nesting Tool – Streamlit
# Features:
# • Large drawing canvas (resizable)
# • Live dimensions display while dragging (rectangles)
# • Add single shape OR queue multiple and “Add All”
# • L-shape wizard with default 90° split at the inside corner
# • Cutouts (sink/cooktop) stored under parent shape
# • Polygon-aware: store polygon vertices; placeholder for nesting
# • Robust session_state to avoid “not adding to parts list” bugs

import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageColor

# -------------------------- Page setup --------------------------
st.set_page_config(page_title="Nesting Tool", layout="wide")

# -------------------------- Helpers & Data Models --------------------------
@dataclass
class Cutout:
    kind: str  # 'sink' | 'cooktop' | 'other'
    x: float   # top-left x in same units as parent (inches)
    y: float   # top-left y
    w: float
    h: float

@dataclass
class RectShape:
    shape_id: str
    label: str
    w: float
    h: float
    qty: int
    note: str = ""
    cutouts: List[Cutout] = None

@dataclass
class LShape:
    shape_id: str
    label: str
    outer_w: float
    outer_h: float
    notch_w: float
    notch_h: float
    qty: int
    # auto-split yields two rectangles
    note: str = "L-shape (auto split 90° at inside corner)"

@dataclass
class PolyShape:
    shape_id: str
    label: str
    points: List[Tuple[float, float]]  # in inches
    qty: int
    note: str = "Polygon part"

def init_state():
    ss = st.session_state
    ss.setdefault("ppi", 8.0)  # pixels per inch (scale)
    ss.setdefault("parts", [])  # committed parts list (Rect/L/Poly normalized to rows)
    ss.setdefault("pending", [])  # shapes queued from canvas before commit
    ss.setdefault("selected_parent_for_cutout", None)
    ss.setdefault("last_drawn_dims", None)  # (w,h) live display
    ss.setdefault("shape_counter", 1)
    ss.setdefault("canvas_key_bump", 0)  # force rerender when needed

init_state()

def px_to_in(val_px: float, ppi: float) -> float:
    return round(val_px / ppi, 3)

def in_to_px(val_in: float, ppi: float) -> int:
    return int(round(val_in * ppi))

def next_shape_id() -> str:
    st.session_state.shape_counter += 1
    return f"S{st.session_state.shape_counter:04d}"

def add_rect_to_pending(label: str, w: float, h: float, qty: int, note: str = ""):
    st.session_state.pending.append(
        {"type": "rect", "data": asdict(RectShape(next_shape_id(), label, w, h, qty, note, []))}
    )

def add_lshape_to_pending(label: str, outer_w: float, outer_h: float, notch_w: float, notch_h: float, qty: int):
    st.session_state.pending.append(
        {"type": "lshape", "data": asdict(LShape(next_shape_id(), label, outer_w, outer_h, notch_w, notch_h, qty))}
    )

def add_poly_to_pending(label: str, points_in: List[Tuple[float, float]], qty: int, note: str = "Polygon part"):
    st.session_state.pending.append(
        {"type": "poly", "data": asdict(PolyShape(next_shape_id(), label, points_in, qty, note))}
    )

def split_lshape_into_rects(outer_w, outer_h, notch_w, notch_h) -> List[Tuple[float, float]]:
    """
    Default split: 90° at inside corner of the L.
    We return two rectangles that tile the L. A simple consistent choice:
    - Rect A: (outer_w - notch_w) by outer_h
    - Rect B: notch_w by (outer_h - notch_h)
    """
    a_w = max(outer_w - notch_w, 0)
    a_h = outer_h
    b_w = notch_w
    b_h = max(outer_h - notch_h, 0)
    rects = []
    if a_w > 0 and a_h > 0:
        rects.append((a_w, a_h))
    if b_w > 0 and b_h > 0:
        rects.append((b_w, b_h))
    return rects

def commit_pending_to_parts():
    """
    Normalize all pending shapes into rows for the parts table.
    - Rect: one row
    - L-shape: split to two rect rows with a note
    - Poly: store as polygon row with area
    """
    for item in st.session_state.pending:
        t = item["type"]
        d = item["data"]
        if t == "rect":
            area = d["w"] * d["h"]
            st.session_state.parts.append({
                "ID": d["shape_id"],
                "Type": "Rect",
                "Label": d["label"],
                "W (in)": d["w"],
                "H (in)": d["h"],
                "Qty": d["qty"],
                "Area (in²)": round(area, 2),
                "Note": d.get("note",""),
                "Cutouts": d.get("cutouts", [])
            })
        elif t == "lshape":
            rects = split_lshape_into_rects(d["outer_w"], d["outer_h"], d["notch_w"], d["notch_h"])
            # Add each rectangle as a separate row with same ID but suffixed
            for idx, (w, h) in enumerate(rects, start=1):
                area = w*h
                st.session_state.parts.append({
                    "ID": f'{d["shape_id"]}-{idx}',
                    "Type": "Rect (from L)",
                    "Label": d["label"],
                    "W (in)": w,
                    "H (in)": h,
                    "Qty": d["qty"],
                    "Area (in²)": round(area, 2),
                    "Note": f'L-split: outer {d["outer_w"]}x{d["outer_h"]} - notch {d["notch_w"]}x{d["notch_h"]}'
                })
        elif t == "poly":
            # Approx area via polygon shoelace (in²)
            pts = d["points"]
            area = polygon_area(pts)
            st.session_state.parts.append({
                "ID": d["shape_id"],
                "Type": "Polygon",
                "Label": d["label"],
                "W (in)": "-",
                "H (in)": "-",
                "Qty": d["qty"],
                "Area (in²)": round(area, 2),
                "Note": d.get("note",""),
                "Points": pts
            })
    st.session_state.pending = []

def polygon_area(points: List[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    s = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i+1) % len(points)]
        s += x1*y2 - x2*y1
    return abs(s)/2.0

def add_cutout_to_parent(parent_id: str, cutout: Cutout):
    """
    Adds a cutout to a pending RECT shape (by its shape_id).
    """
    for item in st.session_state.pending:
        if item["type"] == "rect" and item["data"]["shape_id"] == parent_id:
            if item["data"].get("cutouts") is None:
                item["data"]["cutouts"] = []
            item["data"]["cutouts"].append(asdict(cutout))
            return True
    return False

# -------------------------- Sidebar Controls --------------------------
with st.sidebar:
    st.header("Global Settings")
    st.caption("Units assumed: inches. Adjust canvas scale (pixels-per-inch) for dimension accuracy.")
    st.session_state.ppi = st.slider("Canvas scale (pixels per inch)", min_value=4.0, max_value=20.0, value=st.session_state.ppi, step=0.5)
    canvas_w = st.slider("Canvas width (px)", 800, 1800, 1200, 50)
    canvas_h = st.slider("Canvas height (px)", 500, 1200, 700, 25)

    st.divider()
    st.subheader("Slab / Nesting (preview)")
    slab_w = st.number_input("Slab width (in)", min_value=20.0, value=126.0, step=1.0)
    slab_h = st.number_input("Slab height (in)", min_value=20.0, value=63.0, step=1.0)
    kerf = st.number_input("Kerf / spacing (in)", min_value=0.0, value=0.25, step=0.05)
    edge_margin = st.number_input("Edge margin (in)", min_value=0.0, value=1.0, step=0.25)

# -------------------------- Layout --------------------------
left, mid, right = st.columns([1.2, 1.1, 1])

with left:
    st.subheader("Draw Shapes")
    st.caption("Use the toolbar to draw rectangles or points for polygons. Live rectangle W×H appears below while dragging.")
    # Canvas
    canvas_result = st_canvas(
        fill_color="rgba(0, 151, 255, 0.2)",
        stroke_width=2,
        stroke_color="#0097FF",
        background_color="#ffffff",
        update_streamlit=True,              # Important: updates each drag
        height=canvas_h,
        width=canvas_w,
        drawing_mode="rect",                # default; user can switch via UI below
        key=f"canvas_{st.session_state.canvas_key_bump}"
    )

    # Controls for drawing mode
    drawing_mode = st.radio(
        "Drawing mode",
        options=["Rectangle", "Polygon (click points)", "Transform/Select"],
        horizontal=True
    )

    # If user changes mode, bump canvas key to force proper internal mode reset
    desired = {"Rectangle": "rect", "Polygon (click points)": "point", "Transform/Select": "transform"}[drawing_mode]
    # No direct prop to set mode dynamically via st_canvas API outside rerender; we force rerender only when switching.
    # This is a light workaround—clicking the radio will refresh the app anyway.

    # Live rectangle dimensions while dragging
    live_text_placeholder = st.empty()
    if canvas_result.json_data is not None and "objects" in canvas_result.json_data:
        objs = canvas_result.json_data["objects"]
        # Find the most recently added rectangle-like object and show its live dimensions
        if len(objs) > 0:
            last = objs[-1]
            # Fabric.js rectangle has "width" & "height" in px (scaled by "scaleX"/"scaleY")
            if last.get("type") == "rect":
                w_px = float(last.get("width", 0)) * float(last.get("scaleX", 1))
                h_px = float(last.get("height", 0)) * float(last.get("scaleY", 1))
                w_in = px_to_in(w_px, st.session_state.ppi)
                h_in = px_to_in(h_px, st.session_state.ppi)
                st.session_state.last_drawn_dims = (w_in, h_in)
                live_text_placeholder.info(f"Live rectangle size: **{w_in:.2f} in × {h_in:.2f} in**")
            elif last.get("type") == "circle":
                r_px = float(last.get("radius", 0)) * float(last.get("scaleX", 1))
                d_in = px_to_in(2*r_px, st.session_state.ppi)
                live_text_placeholder.info(f"Live circle diameter: **{d_in:.2f} in**")
            elif last.get("type") in ("polygon", "polyline", "path"):
                live_text_placeholder.info("Live polygon: points updating...")
    else:
        live_text_placeholder.empty()

    st.markdown("—")
    st.caption("**Tip:** After drawing a rectangle, use the ‘Add this shape’ panel to capture dimensions and queue it.")

    # Add-from-canvas helpers
    with st.expander("Add this shape (from last drawn rectangle)", expanded=True):
        colA, colB, colC = st.columns([1, 1, 1])
        with colA:
            rect_label = st.text_input("Label", value="Countertop Part")
        with colB:
            rect_qty = st.number_input("Qty", min_value=1, value=1, step=1)
        with colC:
            rect_note = st.text_input("Note", value="")

        # Auto-read the last rectangle dims if present
        auto_dims = st.session_state.last_drawn_dims or (0.0, 0.0)
        col1, col2 = st.columns(2)
        with col1:
            w_in = st.number_input("Width (in)", min_value=0.0, value=float(auto_dims[0]), step=0.25, format="%.3f")
        with col2:
            h_in = st.number_input("Height (in)", min_value=0.0, value=float(auto_dims[1]), step=0.25, format="%.3f")

        if st.button("➕ Add this rectangle to pending", type="primary"):
            if w_in > 0 and h_in > 0:
                add_rect_to_pending(rect_label.strip() or "Countertop Part", w_in, h_in, rect_qty, rect_note)
                st.success(f"Added {w_in:.2f}×{h_in:.2f} (Qty {rect_qty}) to pending.")
            else:
                st.warning("Width and Height must be > 0.")

    with st.expander("Add Polygon (click points on canvas, then capture)", expanded=False):
        pcol1, pcol2 = st.columns([2,1])
        with pcol1:
            poly_label = st.text_input("Polygon label", value="Polygon Part")
        with pcol2:
            poly_qty = st.number_input("Polygon qty", min_value=1, value=1, step=1)
        if st.button("Capture polygon from canvas objects"):
            pts_in = extract_polygon_points_in_inches(canvas_result, st.session_state.ppi)
            if len(pts_in) >= 3:
                add_poly_to_pending(poly_label.strip() or "Polygon Part", pts_in, poly_qty)
                st.success(f"Captured polygon with {len(pts_in)} points.")
            else:
                st.warning("Draw a polygon (as series of points/shape) before capturing.")

with mid:
    st.subheader("Shape Wizards")
    with st.expander("L-Shape (default 90° split at inside corner)", expanded=True):
        l1, l2 = st.columns(2)
        with l1:
            l_label = st.text_input("L label", value="L-Shape")
            l_outer_w = st.number_input("Outer width (in)", min_value=0.0, value=80.0, step=0.5)
            l_outer_h = st.number_input("Outer height (in)", min_value=0.0, value=26.0, step=0.5)
        with l2:
            l_notch_w = st.number_input("Notch width (in)", min_value=0.0, value=24.0, step=0.5)
            l_notch_h = st.number_input("Notch height (in)", min_value=0.0, value=18.0, step=0.5)
            l_qty = st.number_input("Qty", min_value=1, value=1, step=1)
        if st.button("➕ Add L-Shape to pending"):
            # sanity: notch must fit in outer dims
            if l_notch_w < l_outer_w and l_notch_h < l_outer_h:
                add_lshape_to_pending(l_label.strip() or "L-Shape", l_outer_w, l_outer_h, l_notch_w, l_notch_h, l_qty)
                st.success("L-Shape queued (will auto-split to two rectangles on commit).")
            else:
                st.error("Notch must be smaller than outer dimensions.")

    with st.expander("Cutouts (sink / cooktop) for a pending rectangle", expanded=True):
        pending_rects = [(p["data"]["shape_id"], f'{p["data"]["shape_id"]} – {p["data"]["label"]} {p["data"]["w"]}x{p["data"]["h"]}')
                         for p in st.session_state.pending if p["type"] == "rect"]
        if len(pending_rects) == 0:
            st.caption("No pending rectangles yet. Add a rectangle first.")
        else:
            pid, display = None, None
            selection = st.selectbox("Select parent rectangle", options=[x[1] for x in pending_rects])
            # map back to id
            for sid, disp in pending_rects:
                if disp == selection:
                    pid = sid
                    display = disp
                    break
            c1, c2 = st.columns(2)
            with c1:
                cut_kind = st.selectbox("Cutout type", options=["sink", "cooktop", "other"], index=0)
                cut_w = st.number_input("Cutout width (in)", min_value=0.0, value=33.0, step=0.25)
                cut_h = st.number_input("Cutout height (in)", min_value=0.0, value=22.0, step=0.25)
            with c2:
                cut_x = st.number_input("Offset X from parent left (in)", min_value=0.0, value=10.0, step=0.25)
                cut_y = st.number_input("Offset Y from parent top (in)", min_value=0.0, value=10.0, step=0.25)

            if st.button("➕ Add cutout to selected parent"):
                ok = add_cutout_to_parent(pid, Cutout(cut_kind, cut_x, cut_y, cut_w, cut_h))
                if ok:
                    st.success(f"Cutout added to {display}")
                else:
                    st.error("Could not add cutout. Make sure a pending rectangle is selected.")

    st.markdown("---")
    colq1, colq2 = st.columns([1,1])
    with colq1:
        if st.button("✅ Add ALL pending to parts list (commit)", type="primary"):
            if len(st.session_state.pending) == 0:
                st.info("Nothing pending to add.")
            else:
                commit_pending_to_parts()
                st.success("Pending shapes committed to parts list.")
    with colq2:
        if st.button("🗑️ Clear pending (not committed)"):
            st.session_state.pending = []
            st.info("Pending cleared.")

    # Show pending preview
    if len(st.session_state.pending) > 0:
        st.caption("Pending (not yet added):")
        st.dataframe(pd.DataFrame([
            {
                "Type": p["type"],
                "Label": p["data"].get("label") or p["data"].get("Label"),
                "Dims/Points": (
                    f'{p["data"].get("w","-")}×{p["data"].get("h","-")}'
                    if p["type"] == "rect" else
                    f'L: {p["data"]["outer_w"]}×{p["data"]["outer_h"]} (notch {p["data"]["notch_w"]}×{p["data"]["notch_h"]})'
                    if p["type"] == "lshape" else
                    f'{len(p["data"]["points"])} pts'
                ),
                "Qty": p["data"].get("qty", 1),
                "Note": p["data"].get("note","")
            } for p in st.session_state.pending
        ]), use_container_width=True)

with right:
    st.subheader("Parts List")
    if len(st.session_state.parts) == 0:
        st.caption("No parts yet. Add some from the left.")
    else:
        parts_df = pd.DataFrame(st.session_state.parts)
        st.dataframe(parts_df, use_container_width=True)

        st.download_button(
            "⬇️ Download parts as CSV",
            data=parts_df.to_csv(index=False),
            file_name="parts_list.csv",
            mime="text/csv"
        )

    st.markdown("---")
    st.subheader("Simple Nesting Preview (greedy)")
    st.caption("Very basic first-fit on a single slab; respects kerf and margins. For visualization only.")
    if st.button("Run nesting preview"):
        if len(st.session_state.parts) == 0:
            st.info("No parts to nest.")
        else:
            layout = simple_nest(st.session_state.parts, slab_w, slab_h, kerf, edge_margin)
            st.pyplot(draw_layout_matplotlib(layout, slab_w, slab_h))  # draws a quick preview

# -------------------------- Canvas Utilities --------------------------
def extract_polygon_points_in_inches(canvas_result, ppi: float) -> List[Tuple[float, float]]:
    """
    Try to pull a polygon-like set of points from canvas objects.
    Supports:
    - A 'polygon' object with .points
    - A series of 'circle' points (user placed)
    Falls back to empty list if not found.
    """
    out: List[Tuple[float, float]] = []
    jd = canvas_result.json_data
    if not jd or "objects" not in jd:
        return out
    objs = jd["objects"]
    # Try explicit polygon
    for obj in objs:
        if obj.get("type") == "polygon":
            pts = obj.get("points") or []
            for p in pts:
                x_in = px_to_in(float(p.get("x", 0)), ppi)
                y_in = px_to_in(float(p.get("y", 0)), ppi)
                out.append((x_in, y_in))
            if len(out) >= 3:
                return out
    # Fallback: points from circles (click points)
    pts_tmp = []
    for obj in objs:
        if obj.get("type") == "circle":
            cx = float(obj.get("left", 0)) + float(obj.get("radius", 0)) * float(obj.get("scaleX", 1))
            cy = float(obj.get("top", 0))  + float(obj.get("radius", 0)) * float(obj.get("scaleY", 1))
            pts_tmp.append((px_to_in(cx, ppi), px_to_in(cy, ppi)))
    if len(pts_tmp) >= 3:
        return pts_tmp
    return out

# -------------------------- Simple Nesting (Preview) --------------------------
def simple_nest(parts_rows: List[Dict], slab_w: float, slab_h: float, kerf: float, margin: float):
    """
    Very naive first-fit shelf algorithm for Rect* parts only.
    Ignores polygons and cutouts for the preview.
    Returns a list of placed rectangles: {x,y,w,h,label}
    """
    # Build a flat list of rectangles by Qty
    rects = []
    for row in parts_rows:
        if row.get("Type","").startswith("Rect"):
            qty = int(row.get("Qty", 1))
            w = float(row.get("W (in)", 0))
            h = float(row.get("H (in)", 0))
            for _ in range(qty):
                rects.append({"w": w, "h": h, "label": row.get("Label", "")})
    # Sort largest-first by height, then width
    rects.sort(key=lambda r: (max(r["w"], r["h"]), min(r["w"], r["h"])), reverse=True)

    placements = []
    cursor_x = margin
    cursor_y = margin
    shelf_h = 0.0

    usable_w = slab_w - 2*margin
    usable_h = slab_h - 2*margin

    for r in rects:
        w, h = r["w"], r["h"]
        # Try w×h, else try rotated
        placed = False
        for (rw, rh) in [(w,h), (h,w)]:
            if rw <= usable_w - (cursor_x - margin) and rh <= usable_h - (cursor_y - margin):
                placements.append({"x": cursor_x, "y": cursor_y, "w": rw, "h": rh, "label": r["label"]})
                cursor_x += rw + kerf
                shelf_h = max(shelf_h, rh)
                placed = True
                break
        if not placed:
            # new shelf
            cursor_x = margin
            cursor_y += shelf_h + kerf
            shelf_h = 0.0
            # try again on new shelf
            for (rw, rh) in [(w,h), (h,w)]:
                if (cursor_y - margin + rh) <= usable_h and rw <= usable_w:
                    placements.append({"x": cursor_x, "y": cursor_y, "w": rw, "h": rh, "label": r["label"]})
                    cursor_x += rw + kerf
                    shelf_h = max(shelf_h, rh)
                    placed = True
                    break
        # If still not placed, it simply won't fit in this naive preview
    return placements

def draw_layout_matplotlib(placements: List[Dict], slab_w: float, slab_h: float):
    """
    Render the slab and placed rectangles with matplotlib (no specific colors).
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    # slab
    ax.add_patch(plt.Rectangle((0,0), slab_w, slab_h, fill=False, linewidth=2))
    # parts
    for p in placements:
        ax.add_patch(plt.Rectangle((p["x"], p["y"]), p["w"], p["h"], fill=False))
        ax.text(p["x"] + p["w"]/2, p["y"] + p["h"]/2, p["label"], ha='center', va='center', fontsize=8)
    ax.set_xlim(0, slab_w)
    ax.set_ylim(0, slab_h)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()  # canvas y grows downward; invert to visualize like screen
    ax.set_xlabel("inches")
    ax.set_ylabel("inches")
    fig.tight_layout()
    return fig
