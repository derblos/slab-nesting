# app.py — Slab/Sheet Nesting MVP (rectangles) + "Sketch → CSV (beta)"
# - Packs rectangles with relief, kerf spacing, oversize per side.
# - 0°/90° rotations only; multi-split when necessary (with seam indicators).
# - Tries all existing sheets before opening a new one.
# - SVG labels include gross WxH and net WxH.
# - NEW: Upload a hand-drawn sketch → OCR dimension candidates → build parts → insert into CSV.

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math
import pandas as pd
from io import StringIO
import streamlit as st

# --- NEW: OCR imports ---
import numpy as np
from PIL import Image
import pytesseract
import cv2

# ───────────────────────── Data models ─────────────────────────

@dataclass
class Sheet:
    id: str
    w: float  # usable width (X)
    h: float  # usable height (Y)
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
                return None
            can = min(sw - (curr_x - sx), remain)
            top = max(top, sy)
            if top + H > self.sheet.h + 1e-9:
                return None
            remain -= can
            curr_x += can
            j += 1
        return x, top

    def try_place(self, w: float, h: float) -> Optional[Tuple[float, float]]:
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
        candidates.append((L, D, 0))
    if allow_rotate and L <= usable_h + 1e-9:
        candidates.append((D, L, 90))
    return candidates

def multi_split_lengths(total_len: float, usable_w: float, min_seg: float) -> List[float]:
    k_min = math.ceil(total_len / usable_w)
    k_max = math.floor(total_len / min_seg)
    if k_min > k_max or k_min <= 0:
        raise ValueError(f"Infeasible split: length {total_len} cannot be divided with min {min_seg} and max {usable_w}.")
    k = k_min
    segs = [total_len / k for _ in range(k)]
    for i in range(k):
        segs[i] = max(min(segs[i], usable_w), min_seg)
    diff = total_len - sum(segs)
    iters = 0
    while abs(diff) > 1e-6 and iters < 10000:
        iters += 1
        moved = 0.0
        for j in range(k):
            if diff > 0:
                room = usable_w - segs[j]
                delta = min(room, diff)
                if delta > 0:
                    segs[j] += delta; diff -= delta; moved += delta
            else:
                room = segs[j] - min_seg
                delta = min(room, -diff)
                if delta > 0:
                    segs[j] -= delta; diff += delta; moved += delta
        if moved == 0.0:
            break
    if any(s < min_seg - 1e-5 or s > usable_w + 1e-5 for s in segs):
        raise ValueError("Split adjustment failed to respect bounds.")
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
        if D0 > L0:
            L0, D0 = D0, L0
        for _ in range(req.qty):
            cands = choose_orientation_candidates(L0, D0, usable_w, usable_h, allow_rotate=allow_rotate)
            if not cands:
                raise ValueError(f"'{req.id}' ({L0}×{D0}) violates depth rule for all orientations (usable depth {usable_h}).")
            cands.sort(key=lambda t: t[1])
            chosen = None
            for w_len, h_dep, rot in cands:
                if rot == 0 and w_len <= usable_w + 1e-9:
                    chosen = ("nosplit", rot, w_len, h_dep); break
                if rot == 90 and h_dep <= usable_w + 1e-9:
                    chosen = ("nosplit", rot, w_len, h_dep); break
            if chosen and chosen[0] == "nosplit":
                _, rot, L_axis_w, D_axis_h = chosen
                net_w = (L_axis_w if rot == 0 else D_axis_h)
                net_h = (D_axis_h if rot == 0 else L_axis_w)
                w = net_w + 2*oversize_per_side
                h = net_h + 2*oversize_per_side
                out.append(Part(req.id, w, h, net_w, net_h, (0, 90) if allow_rotate else (0,), None, None, None, False, False))
                continue
            w_len, h_dep, rot_for_seams = cands[0]
            length_to_split = w_len if rot_for_seams == 0 else h_dep
            segs = multi_split_lengths(length_to_split, usable_w, min_seam_offset)
            k = len(segs)
            for i_seg, seg_len in enumerate(segs, start=1):
                if rot_for_seams == 0:
                    net_w, net_h = seg_len, h_dep
                else:
                    net_w, net_h = w_len, seg_len
                w = net_w + 2*oversize_per_side
                h = net_h + 2*oversize_per_side
                out.append(Part(f"{req.id}-S{i_seg}", w, h, net_w, net_h, (0, 90) if allow_rotate else (0,),
                                req.id, i_seg, k, i_seg>1, i_seg<k))
    return out

# ───────────────────────── Packing across sheets ─────────────────────────

def pack_rectangles_across_sheets(
    usable_w: float, usable_h: float, parts: List[Part], spacing: float,
) -> Tuple[List[Placed], List[Sheet]]:
    items = list(parts)
    items.sort(key=lambda p: p.w * p.h, reverse=True)
    placements: List[Placed] = []
    sheets: List[Sheet] = []
    skylines: List[Skyline] = []

    def new_sheet() -> Skyline:
        s = Sheet(id=f"Sheet-{len(sheets)+1}", w=usable_w, h=usable_h, index=len(sheets)+1)
        sheets.append(s)
        sk = Skyline(s, spacing=spacing)
        skylines.append(sk)
        return sk

    if not skylines:
        new_sheet()

    for p in items:
        placed = False
        orientations = [(0, p.w, p.h), (90, p.h, p.w)] if 90 in p.rot_allowed else [(0, p.w, p.h)]
        orientations.sort(key=lambda x: x[2])
        for sk in skylines:
            for rot, w, h in orientations:
                if h > usable_h + 1e-9: continue
                pos = sk.try_place(w, h)
                if pos:
                    x, y = pos
                    placements.append(Placed(p.id, x, y, w, h, p.net_w, p.net_h, rot, sk.sheet.id,
                                             p.group_id, p.seg_idx, p.seg_total, p.seam_before, p.seam_after))
                    placed = True
                    break
            if placed: break
        if not placed:
            sk = new_sheet()
            success = False
            for rot, w, h in orientations:
                if h > usable_h + 1e-9: continue
                pos = sk.try_place(w, h)
                if pos:
                    x, y = pos
                    placements.append(Placed(p.id, x, y, w, h, p.net_w, p.net_h, rot, sk.sheet.id,
                                             p.group_id, p.seg_idx, p.seg_total, p.seam_before, p.seam_after))
                    success = True
                    break
            if not success:
                raise RuntimeError(f"Failed to place part {p.id} on a fresh sheet.")
    return placements, sheets

# ───────────────────────── SVG output ─────────────────────────

def _fmt(n: float) -> str:
    return f"{n:.3f}".rstrip('0').rstrip('.')

def _seam_lines(pl: Placed) -> List[str]:
    inset = 1.0
    dash = 'stroke="red" stroke-width="0.8" stroke-dasharray="3,2"'
    lines = []
    if pl.rot == 0:
        if pl.seam_before:
            x = pl.x + inset
            lines.append(f'<line x1="{x}" y1="{pl.y+inset}" x2="{x}" y2="{pl.y+pl.h-inset}" {dash}/>')
        if pl.seam_after:
            x = pl.x + pl.w - inset
            lines.append(f'<line x1="{x}" y1="{pl.y+inset}" x2="{x}" y2="{pl.y+pl.h-inset}" {dash}/>')
    else:
        if pl.seam_before:
            y = pl.y + inset
            lines.append(f'<line x1="{pl.x+inset}" y1="{y}" x2="{pl.x+pl.w-inset}" y2="{y}" {dash}/>')
        if pl.seam_after:
            y = pl.y + pl.h - inset
            lines.append(f'<line x1="{pl.x+inset}" y1="{y}" x2="{pl.x+pl.w-inset}" y2="{y}" {dash}/>')
    return lines

def _label_for(pl: Placed) -> List[str]:
    label = pl.part_id
    if pl.seg_total and pl.seg_total > 1:
        label += f" [{pl.seg_idx}/{pl.seg_total}]"
    line1 = f'{label} ({pl.rot}°)'
    line2 = f'{_fmt(pl.w)}×{_fmt(pl.h)} (net {_fmt(pl.net_w)}×{_fmt(pl.net_h)})'
    return [line1, line2]

def to_svg_pages(placements: List[Placed], sheets: List[Sheet]) -> List[str]:
    by: Dict[str, List[Placed]] = {}
    for pl in placements:
        by.setdefault(pl.sheet_id, []).append(pl)
    pages: List[str] = []
    sheet_map = {s.id: s for s in sheets}
    for sid, items in by.items():
        s = sheet_map[sid]
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{s.w}" height="{s.h}" viewBox="0 0 {s.w} {s.h}">']
        out.append(f'<rect x="0" y="0" width="{s.w}" height="{s.h}" fill="none" stroke="black" stroke-width="0.8"/>')
        for r in items:
            out.append(f'<rect x="{r.x}" y="{r.y}" width="{r.w}" height="{r.h}" fill="none" stroke="blue" stroke-width="0.8"/>')
            l1, l2 = _label_for(r)
            out.append(f'<text x="{r.x + 2}" y="{r.y + 8}" font-size="6">{l1}</text>')
            out.append(f'<text x="{r.x + 2}" y="{r.y + 14}" font-size="6">{l2}</text>')
            out.extend(_seam_lines(r))
        out.append("</svg>")
        pages.append("\n".join(out))
    return pages

def to_svg_combined(placements: List[Placed], sheets: List[Sheet], gap: float = 10.0) -> str:
    by: Dict[str, List[Placed]] = {}
    for pl in placements:
        by.setdefault(pl.sheet_id, []).append(pl)
    ordered = sorted(sheets, key=lambda s: s.index)
    max_w = max((s.w for s in ordered), default=0.0)
    total_h = sum((s.h for s in ordered)) + gap * max(0, len(ordered) - 1)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{max_w}" height="{total_h}" viewBox="0 0 {max_w} {total_h}">']
    yoff = 0.0
    for s in ordered:
        out.append(f'<g transform="translate(0,{yoff})">')
        out.append(f'<rect x="0" y="0" width="{s.w}" height="{s.h}" fill="none" stroke="black" stroke-width="0.8"/>')
        for r in by.get(s.id, []):
            out.append(f'<rect x="{r.x}" y="{r.y}" width="{r.w}" height="{r.h}" fill="none" stroke="blue" stroke-width="0.8"/>')
            l1, l2 = _label_for(r)
            out.append(f'<text x="{r.x + 2}" y="{r.y + 8}" font-size="6">{l1}</text>')
            out.append(f'<text x="{r.x + 2}" y="{r.y + 14}" font-size="6">{l2}</text>')
            out.extend(_seam_lines(r))
        out.append("</g>")
        yoff += s.h + gap
    out.append("</svg>")
    return "\n".join(out)

def utilization(placements: List[Placed], sheets: List[Sheet]) -> Tuple[float, float, float]:
    used_area = sum(p.w * p.h for p in placements)
    total_area = sum(s.w * s.h for s in sheets)
    util = (used_area / total_area * 100.0) if total_area > 0 else 0.0
    return used_area, total_area, util

# ───────────────────────── OCR helpers (Sketch → CSV) ─────────────────────────

FRACTION_MAP = {'½': (1,2), '¼': (1,4), '¾': (3,4), '⅛': (1,8), '⅜': (3,8), '⅝': (5,8), '⅞': (7,8)}

def parse_inches_token(tok: str) -> Optional[float]:
    """Parse '37 1/2', '25-1/2', '25.5', '25"', '25 ½' → float inches."""
    t = tok.strip().lower().replace('"','').replace('in','').replace('”','').replace('“','')
    # whole number + unicode fraction (e.g., 25½)
    for uf, (a,b) in FRACTION_MAP.items():
        if uf in t:
            try:
                whole = float(t.replace(uf,'').strip() or 0)
                return whole + a/b
            except: pass
    # pattern like 25 1/2 or 25-1/2
    import re
    m = re.match(r'^\s*(\d+(?:\.\d+)?)\s*(?:-| )\s*(\d+)\s*[/⁄]\s*(\d+)\s*$', t)
    if m:
        whole = float(m.group(1)); num=int(m.group(2)); den=int(m.group(3))
        if den != 0: return whole + num/den
    # just a fraction 1/2
    m = re.match(r'^\s*(\d+)\s*[/⁄]\s*(\d+)\s*$', t)
    if m:
        num=int(m.group(1)); den=int(m.group(2))
        if den!=0: return num/den
    # plain float/int
    try:
        return float(t)
    except:
        return None

def extract_dimensions_from_image(file_bytes: bytes) -> List[float]:
    """Return a de-duplicated sorted list of inch values OCR'd from the image."""
    # load
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return []
    # preprocess: rotate if tall, grayscale, resize, threshold
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 5)
    config = "--psm 6"
    text = pytesseract.image_to_string(th, config=config)
    # parse tokens
    raw_tokens = [t for t in text.replace('\n',' ').split(' ') if t.strip()]
    vals = []
    for t in raw_tokens:
        v = parse_inches_token(t)
        if v is not None and 1 <= v <= 300:  # sanity window
            vals.append(round(v, 3))
    # de-dup with tolerance
    uniq = []
    for v in sorted(vals):
        if not uniq or abs(uniq[-1]-v) > 0.01:
            uniq.append(v)
    return uniq

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
    "• Length = **long** dimension (pre-oversize). Depth = **short** dimension (pre-oversize). "
    "If the part’s length exceeds usable width in all allowed orientations, it will be split into the **fewest** segments "
    "between [Min seam offset, usable width]. Seam edges show as **dashed red**. Labels show **gross** and **net** sizes."
)

# keep the CSV box in session state so the Sketch tool can write into it
example_csv = "id,length,depth,qty\nTop-A,62,25.5,2\nSplash,96,4,4\nIsland,132,38,1\nGiant,260,25,1\n"
if "csv_text" not in st.session_state:
    st.session_state["csv_text"] = example_csv
txt = st.text_area("CSV input", key="csv_text", height=200)
df = None
try:
    df = pd.read_csv(StringIO(st.session_state["csv_text"]))
except Exception as e:
    st.error(f"Could not parse CSV: {e}")

if df is not None:
    st.dataframe(df, use_container_width=True)

col1, col2, col3 = st.columns(3)
with col1:
    usable_w = max(0.0, slab_w - 2*relief)
    st.metric("Usable width (X)", f"{usable_w:.3f} {units}")
with col2:
    usable_h = max(0.0, slab_h - 2*relief)
    st.metric("Usable depth (Y)", f"{usable_h:.3f} {units}")
with col3:
    st.metric("Kerf / spacing", f"{kerf:.3f} {units}")

# ───────────── Sketch → CSV (beta)
with st.expander("🖼️ Sketch → CSV (beta) — upload a hand-drawn photo"):
    up = st.file_uploader("Upload JPG/PNG of your sketch", type=["jpg","jpeg","png"])
    if up is not None:
        st.image(up, caption="Uploaded sketch", use_column_width=True)
        if st.button("Extract dimension numbers from sketch"):
            try:
                nums = extract_dimensions_from_image(up.read())
                if nums:
                    st.success(f"Found numbers (inches): {nums}")
                    st.session_state["ocr_nums"] = nums
                else:
                    st.warning("No dimension-like numbers detected. You can still type them below.")
            except Exception as e:
                st.error(f"OCR failed: {e}\nMake sure Tesseract is installed.")
    nums = st.session_state.get("ocr_nums", [])
    st.caption("Use detected numbers (or type your own) to build parts:")
    with st.form("sketch_to_csv"):
        c1, c2, c3, c4 = st.columns([2,1,1,1])
        pid = c1.text_input("Part ID", value="SketchPart")
        length = c2.selectbox("Length (in)", options=[None]+nums, index=0, key="len_sel")
        length_manual = c2.text_input("or type length", value="")
        depth = c3.selectbox("Depth (in)", options=[None]+nums, index=0, key="dep_sel")
        depth_manual = c3.text_input("or type depth", value="")
        qty = c4.number_input("Qty", min_value=1, value=1, step=1)

        submitted = st.form_submit_button("Add to CSV below")
        if submitted:
            def _to_float(vtxt):
                if vtxt.strip():
                    return float(parse_inches_token(vtxt))
                return None
            L = length if length is not None else _to_float(length_manual or "")
            D = depth if depth is not None else _to_float(depth_manual or "")
            if not L or not D:
                st.error("Please choose or type both Length and Depth.")
            else:
                # append to CSV text
                current = st.session_state["csv_text"].strip()
                if not current.endswith("\n"):
                    current += "\n"
                st.session_state["csv_text"] = current + f"{pid},{L},{D},{qty}\n"
                st.success(f"Added {pid} {L}×{D} ×{qty} to the CSV box above.")

if st.button("Nest parts"):
    try:
        if df is None or df.empty:
            st.stop()
        reqs: List[PartReq] = []
        for _, r in df.iterrows():
            pid = str(r["id"]); L = float(r["length"]); D = float(r["depth"]); Q = int(r["qty"])
            if Q > 0: reqs.append(PartReq(pid, L, D, Q))

        parts = expand_and_transform_parts(reqs, usable_w, usable_h, oversize_per_side, min_seam_offset, allow_rotate)
        placements, sheets = pack_rectangles_across_sheets(usable_w, usable_h, parts, kerf)
        used_area, total_area, util_pct = utilization(placements, sheets)

        st.subheader("Results")
        st.write(f"**Sheets used:** {len(sheets)}")
        st.write(f"**Utilization:** {util_pct:.2f}% (used {used_area:.2f} / {total_area:.2f} {units}²)")

        pages = to_svg_pages(placements, sheets)
        for svg_page in pages:
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
