# app.py — Slab/Sheet Nesting (Rectangles) + On-screen Drawing
# - Relief border reduces usable area
# - Kerf/spacing margin between parts
# - Oversize per side on each part
# - Depth rule (height cannot exceed usable depth)
# - 0°/90° rotations only
# - Split only when necessary; multi-split with min seam offset
# - Packs across ALL existing sheets before opening a new one
# - SVG previews per sheet + combined SVG download
# - Labels show Part/Seg/Rot + gross WxH and net WxH
# - Draw rectangles on a grid to add parts (no typing)
# - Robust Streamlit state handling for the CSV text area (no key conflicts)

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from io import StringIO
import math
import pandas as pd
import streamlit as st

# Drawing widget + simple grid image
try:
    from streamlit_drawable_canvas import st_canvas
    HAVE_CANVAS = True
except Exception:
    HAVE_CANVAS = False
from PIL import Image, ImageDraw

# ───────────────────────── Data models ─────────────────────────

@dataclass
class Sheet:
    id: str
    w: float  # usable width (X)
    h: float  # usable height/depth (Y)
    index: int

@dataclass
class PartReq:
    id: str
    length: float   # long side BEFORE oversize
    depth: float    # short side BEFORE oversize
    qty: int

@dataclass
class Part:
    id: str
    w: float             # gross (cut) width after oversize
    h: float             # gross (cut) height after oversize
    net_w: float         # finished width before oversize
    net_h: float         # finished height before oversize
    rot_allowed: Tuple[int, ...] = (0, 90)
    group_id: Optional[str] = None
    seg_idx: Optional[int] = None
    seg_total: Optional[int] = None
    seam_before: bool = False
    seam_after: bool = False

@dataclass
class Placed:
    part_id: str
    x: float
    y: float
    w: float          # gross (cut) width
    h: float          # gross (cut) height
    net_w: float      # finished width
    net_h: float      # finished height
    rot: int
    sheet_id: str
    group_id: Optional[str]
    seg_idx: Optional[int]
    seg_total: Optional[int]
    seam_before: bool
    seam_after: bool

# ───────────────────────── Skyline packer ─────────────────────────

class Skyline:
    """Bottom-left skyline packer with uniform spacing (kerf) margin."""
    def __init__(self, sheet: Sheet, spacing: float = 0.0):
        self.sheet = sheet
        self.spacing = spacing
        self.lines = [(0.0, 0.0, sheet.w)]  # (x, y, width)

    def _fits_here(self, i: int, W: float, H: float) -> Optional[Tuple[float, float]]:
        x, y, _ = self.lines[i]
        top = y
        curr_x = x
        remain = W
        j = i
        while remain > 1e-9:
            if j >= len(self.lines):
                return None
            sx, sy, sw = self.lines[j]
            if sx > curr_x + 1e-9:
                return None  # gap between segments
            can = min(sw - (curr_x - sx), remain)
            top = max(top, sy)
            if top + H > self.sheet.h + 1e-9:
                return None
            remain -= can
            curr_x += can
            j += 1
        return x, top

    def try_place(self, w: float, h: float) -> Optional[Tuple[float, float]]:
        # Inflate by spacing margin
        W = w + 2*self.spacing
        H = h + 2*self.spacing
        for i in range(len(self.lines)):
            pos = self._fits_here(i, W, H)
            if not pos:
                continue
            bx, by = pos
            new_top = by + H
            x, y, width = self.lines[i]
            pre, post = [], []
            left = x
            if bx > x + 1e-9:
                pre.append((x, y, bx - x))
                left = bx
            need = W
            j = i
            while need > 1e-9 and j < len(self.lines):
                sx, sy, sw = self.lines[j]
                seg_left = max(sx, left)
                seg_can = min(sw - (seg_left - sx), need)
                right = seg_left + seg_can
                pre.append((seg_left, new_top, seg_can))
                left = right
                need -= seg_can
                if right < sx + sw - 1e-9:
                    post.append((right, sy, sx + sw - right))
                    j += 1
                    break
                j += 1
            post.extend(self.lines[j:])
            # merge adjacent
            merged = []
            for seg in pre + post:
                if merged and abs(merged[-1][1] - seg[1]) < 1e-9 and abs(merged[-1][0] + merged[-1][2] - seg[0]) < 1e-9:
                    x0, y0, w0 = merged[-1]
                    merged[-1] = (x0, y0, w0 + seg[2])
                else:
                    merged.append(seg)
            self.lines = merged
            return bx + self.spacing, by + self.spacing
        return None

# ───────────────────────── Helpers ─────────────────────────

def choose_orientation_candidates(L: float, D: float, usable_w: float, usable_h: float, allow_rotate=True):
    candidates = []
    if D <= usable_h + 1e-9:
        candidates.append((L, D, 0))     # 0°: length→w, depth→h
    if allow_rotate and L <= usable_h + 1e-9:
        candidates.append((D, L, 90))    # 90°: length→h, depth→w
    return candidates

def multi_split_lengths(total_len: float, usable_w: float, min_seg: float) -> List[float]:
    k_min = math.ceil(total_len / usable_w)
    k_max = math.floor(total_len / min_seg)
    if k_min > k_max or k_min <= 0:
        raise ValueError(f"Infeasible split: {total_len} cannot be divided with min {min_seg} and max {usable_w}.")
    k = k_min
    segs = [total_len / k for _ in range(k)]
    for i in range(k):
        segs[i] = max(min(segs[i], usable_w), min_seg)
    diff = total_len - sum(segs)
    iters = 0
    while abs(diff) > 1e-6 and iters < 10000:
        iters += 1; moved = 0.0
        for j in range(k):
            if diff > 0:
                room = usable_w - segs[j]; delta = min(room, diff)
                if delta > 0: segs[j]+=delta; diff-=delta; moved+=delta
            else:
                room = segs[j] - min_seg; delta = min(room, -diff)
                if delta > 0: segs[j]-=delta; diff+=delta; moved+=delta
        if moved == 0.0: break
    if any(s < min_seg - 1e-5 or s > usable_w + 1e-5 for s in segs):
        raise ValueError("Split adjustment failed.")
    return segs

def expand_and_transform_parts(
    reqs: List[PartReq],
    usable_w: float,
    usable_h: float,
    oversize_per_side: float,
    min_seam_offset: float,
    allow_rotate: bool = True,
) -> List[Part]:
    out: List[Part] = []
    for req in reqs:
        L0, D0 = float(req.length), float(req.depth)
        if D0 > L0: L0, D0 = D0, L0
        for _ in range(req.qty):
            cands = choose_orientation_candidates(L0, D0, usable_w, usable_h, allow_rotate)
            if not cands:
                raise ValueError(f"'{req.id}' ({L0}×{D0}) violates depth rule for all orientations (usable depth {usable_h}).")
            cands.sort(key=lambda t: t[1])  # prefer smaller height
            chosen = None
            for w_len, h_dep, rot in cands:
                if rot == 0 and w_len <= usable_w + 1e-9: chosen=("nosplit",rot,w_len,h_dep); break
                if rot == 90 and h_dep <= usable_w + 1e-9: chosen=("nosplit",rot,w_len,h_dep); break
            if chosen and chosen[0]=="nosplit":
                _, rot, Lw, Dh = chosen
                net_w = (Lw if rot==0 else Dh); net_h = (Dh if rot==0 else Lw)
                w = net_w + 2*oversize_per_side; h = net_h + 2*oversize_per_side
                out.append(Part(req.id, w, h, net_w, net_h, (0,90) if allow_rotate else (0,), None, None, None, False, False))
                continue
            w_len, h_dep, rot_for_seams = cands[0]
            length_to_split = w_len if rot_for_seams==0 else h_dep
            segs = multi_split_lengths(length_to_split, usable_w, min_seam_offset)
            k = len(segs)
            for i_seg, seg_len in enumerate(segs, start=1):
                if rot_for_seams==0: net_w, net_h = seg_len, h_dep
                else:                net_w, net_h = w_len, seg_len
                w = net_w + 2*oversize_per_side; h = net_h + 2*oversize_per_side
                out.append(Part(f"{req.id}-S{i_seg}", w, h, net_w, net_h, (0,90) if allow_rotate else (0,),
                                req.id, i_seg, k, i_seg>1, i_seg<k))
    return out

# ───────────────────────── Packing across sheets ─────────────────────────

def pack_rectangles_across_sheets(usable_w: float, usable_h: float, parts: List[Part], spacing: float):
    items = sorted(parts, key=lambda p: p.w*p.h, reverse=True)
    placements: List[Placed] = []; sheets: List[Sheet] = []; skylines: List[Skyline] = []

    def new_sheet():
        s = Sheet(id=f"Sheet-{len(sheets)+1}", w=usable_w, h=usable_h, index=len(sheets)+1)
        sheets.append(s); sk = Skyline(s, spacing=spacing); skylines.append(sk); return sk
    if not skylines: new_sheet()

    for p in items:
        placed=False
        orientations = [(0,p.w,p.h),(90,p.h,p.w)] if 90 in p.rot_allowed else [(0,p.w,p.h)]
        orientations.sort(key=lambda x:x[2])
        for sk in skylines:
            for rot,w,h in orientations:
                if h>usable_h+1e-9: continue
                pos = sk.try_place(w,h)
                if pos:
                    x,y=pos; placements.append(Placed(p.id,x,y,w,h,p.net_w,p.net_h,rot,sk.sheet.id,
                                                     p.group_id,p.seg_idx,p.seg_total,p.seam_before,p.seam_after))
                    placed=True; break
            if placed: break
        if not placed:
            sk=new_sheet(); success=False
            for rot,w,h in orientations:
                if h>usable_h+1e-9: continue
                pos=sk.try_place(w,h)
                if pos:
                    x,y=pos; placements.append(Placed(p.id,x,y,w,h,p.net_w,p.net_h,rot,sk.sheet.id,
                                                     p.group_id,p.seg_idx,p.seg_total,p.seam_before,p.seam_after))
                    success=True; break
            if not success: raise RuntimeError(f"Failed to place part {p.id} on a fresh sheet.")
    return placements, sheets

# ───────────────────────── SVG output ─────────────────────────

def _fmt(n: float) -> str:
    return f"{n:.3f}".rstrip('0').rstrip('.')

def _seam_lines(pl: Placed) -> List[str]:
    inset = 1.0; dash='stroke="red" stroke-width="0.8" stroke-dasharray="3,2"'; lines=[]
    if pl.rot==0:
        if pl.seam_before:
            x=pl.x+inset; lines.append(f'<line x1="{x}" y1="{pl.y+inset}" x2="{x}" y2="{pl.y+pl.h-inset}" {dash}/>')
        if pl.seam_after:
            x=pl.x+pl.w-inset; lines.append(f'<line x1="{x}" y1="{pl.y+inset}" x2="{x}" y2="{pl.y+pl.h-inset}" {dash}/>')
    else:
        if pl.seam_before:
            y=pl.y+inset; lines.append(f'<line x1="{pl.x+inset}" y1="{y}" x2="{pl.x+pl.w-inset}" y2="{y}" {dash}/>')
        if pl.seam_after:
            y=pl.y+pl.h-inset; lines.append(f'<line x1="{pl.x+inset}" y1="{y}" x2="{pl.x+pl.w-inset}" y2="{y}" {dash}/>')
    return lines

def _label_for(pl: Placed) -> List[str]:
    label=pl.part_id
    if pl.seg_total and pl.seg_total>1: label+=f" [{pl.seg_idx}/{pl.seg_total}]"
    return [f'{label} ({pl.rot}°)', f'{_fmt(pl.w)}×{_fmt(pl.h)} (net {_fmt(pl.net_w)}×{_fmt(pl.net_h)})']

def to_svg_pages(placements: List[Placed], sheets: List[Sheet]) -> List[str]:
    by: Dict[str,List[Placed]] = {}
    for pl in placements: by.setdefault(pl.sheet_id,[]).append(pl)
    pages=[]; smap={s.id:s for s in sheets}
    for sid,items in by.items():
        s=smap[sid]
        out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{s.w}" height="{s.h}" viewBox="0 0 {s.w} {s.h}">']
        out.append(f'<rect x="0" y="0" width="{s.w}" height="{s.h}" fill="none" stroke="black" stroke-width="0.8"/>')
        for r in items:
            out.append(f'<rect x="{r.x}" y="{r.y}" width="{r.w}" height="{r.h}" fill="none" stroke="blue" stroke-width="0.8"/>')
            l1,l2=_label_for(r)
            out.append(f'<text x="{r.x+2}" y="{r.y+8}" font-size="6">{l1}</text>')
            out.append(f'<text x="{r.x+2}" y="{r.y+14}" font-size="6">{l2}</text>')
            out.extend(_seam_lines(r))
        out.append("</svg>"); pages.append("\n".join(out))
    return pages

def to_svg_combined(placements: List[Placed], sheets: List[Sheet], gap: float = 10.0) -> str:
    by: Dict[str,List[Placed]] = {}
    for pl in placements: by.setdefault(pl.sheet_id,[]).append(pl)
    ordered=sorted(sheets,key=lambda s:s.index)
    max_w=max((s.w for s in ordered), default=0.0)
    total_h=sum((s.h for s in ordered)) + gap * max(0, len(ordered)-1)
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{max_w}" height="{total_h}" viewBox="0 0 {max_w} {total_h}">']
    yoff=0.0
    for s in ordered:
        out.append(f'<g transform="translate(0,{yoff})">')
        out.append(f'<rect x="0" y="0" width="{s.w}" height="{s.h}" fill="none" stroke="black" stroke-width="0.8"/>')
        for r in by.get(s.id,[]):
            out.append(f'<rect x="{r.x}" y="{r.y}" width="{r.w}" height="{r.h}" fill="none" stroke="blue" stroke-width="0.8"/>')
            l1,l2=_label_for(r)
            out.append(f'<text x="{r.x+2}" y="{r.y+8}" font-size="6">{l1}</text>')
            out.append(f'<text x="{r.x+2}" y="{r.y+14}" font-size="6">{l2}</text>')
            out.extend(_seam_lines(r))
        out.append("</g>"); yoff+=s.h+gap
    out.append("</svg>"); return "\n".join(out)

def utilization(placements: List[Placed], sheets: List[Sheet]):
    used = sum(p.w*p.h for p in placements); total = sum(s.w*s.h for s in sheets)
    return used, total, (used/total*100.0 if total>0 else 0.0)

# ───────────────────────── On-screen drawing helpers ─────────────────────────

def round_to_step(x: float, step: float) -> float: return round(x/step)*step

def make_grid_image(width_px: int, height_px: int, ppi: float, grid_in: float = 1.0) -> Image.Image:
    img = Image.new("RGB", (width_px, height_px), "white"); d = ImageDraw.Draw(img)
    step_px = max(1, int(grid_in * ppi))
    x=0
    while x<width_px: d.line([(x,0),(x,height_px)], fill=(230,230,230), width=1); x+=step_px
    y=0
    while y<height_px: d.line([(0,y),(width_px,y)], fill=(230,230,230), width=1); y+=step_px
    d.rectangle([(0,0),(width_px-1,height_px-1)], outline=(0,0,0), width=2); return img

def objects_to_parts(json_data, ppi: float, round_step: float = 1/16) -> List[Tuple[float,float,int]]:
    if not json_data or "objects" not in json_data: return []
    dims=[]
    for obj in json_data["objects"]:
        if obj.get("type")!="rect": continue
        w=float(obj.get("width",0))*float(obj.get("scaleX",1))
        h=float(obj.get("height",0))*float(obj.get("scaleY",1))
        if w<=0 or h<=0: continue
        w_in=round_to_step(w/ppi, round_step); h_in=round_to_step(h/ppi, round_step)
        L,D=(w_in,h_in) if w_in>=h_in else (h_in,w_in)
        dims.append((L,D))
    from collections import Counter
    cnt=Counter(dims)
    def sort_key(item): (L,D),q=item; return (-(L*D), -L, -D)
    return [(L,D,q) for ((L,D),q) in sorted(cnt.items(), key=sort_key)]

# ───────────────────────── Streamlit UI ─────────────────────────

st.set_page_config(page_title="Slab Nesting MVP", layout="wide")
st.title("🧩 Slab/Sheet Nesting (Rectangles)")

with st.sidebar:
    st.header("Global parameters")
    units = st.selectbox("Units", ["in"], index=0)
    slab_w = st.number_input('Slab/Sheet width (e.g., 130")', min_value=1.0, value=130.0, step=0.5)
    slab_h = st.number_input('Slab/Sheet height/depth (e.g., 63")', min_value=1.0, value=63.0, step=0.5)
    relief = st.number_input('Relief per edge (e.g., 1")', min_value=0.0, value=1.0, step=0.125)
    kerf = st.number_input('Kerf / min spacing (e.g., 0.125")', min_value=0.0, value=0.125, step=0.125)
    oversize_per_side = st.number_input('Oversize per side (e.g., 0.0625")', min_value=0.0, value=0.0625, step=0.0625)
    min_seam_offset = st.number_input('Min seam offset (e.g., 24")', min_value=0.0, value=24.0, step=1.0)
    allow_rotate = st.checkbox("Allow 90° rotation", value=True)

st.subheader("Enter parts (Length × Depth × Qty)")
st.caption(
    "• Length = **long** dimension (pre-oversize). Depth = **short** (pre-oversize). "
    "If a part’s length exceeds usable width in all allowed orientations, it will be split into the **fewest** segments "
    "with each segment in [Min seam offset, usable width]. Seam edges show as **dashed red**. "
    "Labels show **gross** (cut) and **net** (finished) sizes."
)

# ---- CSV text area: robust state handling (no widget-key assignment during run)
example_csv = "id,length,depth,qty\nTop-A,62,25.5,2\nSplash,96,4,4\nIsland,132,38,1\nGiant,260,25,1\n"
if "csv_text" not in st.session_state:
    st.session_state["csv_text"] = example_csv
if "csv_text_widget" not in st.session_state:
    st.session_state["csv_text_widget"] = st.session_state["csv_text"]

# Apply any pending programmatic update to the widget value BEFORE rendering it
if "csv_text_widget_new_value" in st.session_state:
    st.session_state["csv_text_widget"] = st.session_state.pop("csv_text_widget_new_value")

csv_text_val = st.text_area(
    "CSV input",
    value=st.session_state["csv_text_widget"],
    key="csv_text_widget",
    height=200,
)
# Sync widget -> storage every run
st.session_state["csv_text"] = csv_text_val

# Parse CSV
df = None
try:
    df = pd.read_csv(StringIO(st.session_state["csv_text"]))
except Exception as e:
    st.error(f"Could not parse CSV: {e}")

if df is not None:
    st.dataframe(df, use_container_width=True)

# Usable dims + kerf
col1, col2, col3 = st.columns(3)
with col1:
    usable_w = max(0.0, slab_w - 2*relief)
    st.metric("Usable width (X)", f"{usable_w:.3f} {units}")
with col2:
    usable_h = max(0.0, slab_h - 2*relief)
    st.metric("Usable depth (Y)", f"{usable_h:.3f} {units}")
with col3:
    st.metric("Kerf / spacing", f"{kerf:.3f} {units}")

# ───────────── On-screen drawing (no typing)
st.subheader("Draw parts (no typing)")
if not HAVE_CANVAS:
    st.info("Install the drawing tool to enable this panel: `pip install streamlit-drawable-canvas`")
else:
    with st.expander("🧱 Draw rectangles on a grid (auto-measured)"):
        u_w = max(0.0, slab_w - 2*relief)
        u_h = max(0.0, slab_h - 2*relief)

        preview_width_px = st.slider("Preview width (pixels)", 400, 1100, 800, 50)
        grid_in = st.selectbox("Grid spacing (inches)", [0.5, 1.0, 2.0], index=1)
        round_step = st.selectbox(
            "Round measured sizes to",
            [1/16, 1/8, 1/4, 1/2, 1.0], index=0,
            format_func=lambda s: f'1/{int(1/s)}"' if s < 1 else '1"'
        )
        preview_ppi = preview_width_px / max(u_w, 1e-6)
        preview_height_px = int(max(u_h, 1e-6) * preview_ppi)

        grid_img = make_grid_image(preview_width_px, preview_height_px, preview_ppi, grid_in=grid_in)

        if "draw_key" not in st.session_state:
            st.session_state.draw_key = 0

        c1, c2 = st.columns([3, 1])
        with c1:
            st.caption("Toolbar: pick rectangle tool, then drag to draw parts. Resize/move to adjust.")
            try:
                canvas_result = st_canvas(
                    fill_color="rgba(0, 0, 255, 0.15)",
                    stroke_width=2,
                    stroke_color="#0000FF",
                    background_color="#FFFFFF",
                    background_image=grid_img,
                    height=preview_height_px,
                    width=preview_width_px,
                    drawing_mode="rect",
                    display_toolbar=True,
                    update_streamlit=True,
                    key=f"canvas-{st.session_state.draw_key}",
                )
            except AttributeError:
                st.warning("Using a plain background for compatibility with your Streamlit version.")
                canvas_result = st_canvas(
                    fill_color="rgba(0, 0, 255, 0.15)",
                    stroke_width=2,
                    stroke_color="#0000FF",
                    background_color="#FFFFFF",
                    background_image=None,
                    height=preview_height_px,
                    width=preview_width_px,
                    drawing_mode="rect",
                    display_toolbar=True,
                    update_streamlit=True,
                    key=f"canvas-{st.session_state.draw_key}",
                )

        with c2:
            if st.button("Clear canvas"):
                st.session_state.draw_key += 1
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()

        parts_from_canvas = objects_to_parts(
            getattr(canvas_result, "json_data", None),
            preview_ppi,
            round_step=round_step,
        )

        if parts_from_canvas:
            st.success(
                f"Detected {sum(q for _, _, q in parts_from_canvas)} rectangle(s) "
                f"in {len(parts_from_canvas)} unique size(s)."
            )
            st.dataframe(
                pd.DataFrame(
                    [{"length": L, "depth": D, "qty": q} for (L, D, q) in parts_from_canvas]
                ),
                use_container_width=True,
            )

            if st.button("➕ Add these to the CSV input above"):
                current = st.session_state["csv_text"].strip()
                if not current.endswith("\n"):
                    current += "\n"
                for idx, (L, D, q) in enumerate(parts_from_canvas, start=1):
                    current += f"Draw-{idx},{L},{D},{q}\n"

                # Update storage, schedule widget update for next run, then rerun
                st.session_state["csv_text"] = current
                st.session_state["csv_text_widget_new_value"] = current
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
        else:
            st.info("Draw a rectangle to get started.")

# ───────────── Nesting run
if st.button("Nest parts"):
    try:
        if df is None or df.empty:
            st.stop()
        reqs: List[PartReq] = []
        for _, r in df.iterrows():
            pid = str(r["id"]); L = float(r["length"]); D = float(r["depth"]); Q = int(r["qty"])
            if Q > 0: reqs.append(PartReq(pid, L, D, Q))

        parts = expand_and_transform_parts(
            reqs,
            usable_w=usable_w,
            usable_h=usable_h,
            oversize_per_side=oversize_per_side,
            min_seam_offset=min_seam_offset,
            allow_rotate=allow_rotate,
        )

        placements, sheets = pack_rectangles_across_sheets(
            usable_w=usable_w,
            usable_h=usable_h,
            parts=parts,
            spacing=kerf,
        )

        used_area, total_area, util_pct = utilization(placements, sheets)
        st.subheader("Results")
        st.write(f"**Sheets used:** {len(sheets)}")
        st.write(f"**Utilization:** {util_pct:.2f}% (used {used_area:.2f} / {total_area:.2f} {units}²)")

        for svg_page in to_svg_pages(placements, sheets):
            st.markdown(svg_page, unsafe_allow_html=True)
            st.divider()

        download_svg = to_svg_combined(placements, sheets)
        st.download_button("Download SVG", data=download_svg, file_name="nest_layouts.svg", mime="image/svg+xml")

        st.subheader("Placed parts")
        table = pd.DataFrame([{
            "sheet": p.sheet_id, "part": p.part_id, "group": p.group_id or "",
            "seg": f"{p.seg_idx}/{p.seg_total}" if p.seg_total else "",
            "x": round(p.x, 3), "y": round(p.y, 3),
            "gross_w": round(p.w, 3), "gross_h": round(p.h, 3),
            "net_w": round(p.net_w, 3), "net_h": round(p.net_h, 3),
            "rot": p.rot, "seam_before": p.seam_before, "seam_after": p.seam_after
        } for p in placements])
        st.dataframe(table, use_container_width=True)

    except Exception as e:
        st.error(str(e))
