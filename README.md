# Slab / Sheet Nesting (Rectangles)

A lightweight Streamlit app that nests **rectangular** parts onto slabs/sheets with shop-friendly rules: relief borders, kerf spacing, per-side oversize, **0°/90° rotations**, and **automatic multi-splits** when a part’s length exceeds usable width. The app packs across **all existing sheets** (better utilization), draws **seam indicators** (dashed red), and labels each piece with **gross (cut)** and **net (finished)** sizes. Exports a **single valid SVG** layout.

> ✅ Built for countertop workflows; future-proofed to swap in irregular (polygon) nesting later.

![App](docs/screenshot-app.png)
![SVG](docs/screenshot-svg.png)

---

## Features

- **Relief border** per slab edge → reduces usable area (e.g., 130"×63" with 1" relief ⇒ **128"×61"**).
- **Kerf as spacing** between parts (e.g., ⅛" = `0.125`).
- **Oversize per side** added to every part before nesting (e.g., 1⁄16" = `0.0625` → +⅛" to both W & H).
- **Depth rule:** placed height (Y) must not exceed usable depth.
- **Rotations:** only **0°/90°** (global toggle).
- **Seams / splits:** only when necessary; automatic **multi-split** into the *fewest* segments; each segment ∈ `[Min seam offset, usable width]`.
- **Pack across all sheets** before opening a new sheet (fills gaps with splashes/shorts).
- **SVG preview & download:** one SVG per sheet on screen; **single combined SVG** for download.
- **Labels:** `Part [i/k] (rot°)` on line 1; `gross WxH (net WxH)` on line 2.
- **Placement table** + utilization stats.

---

## Quickstart

### Requirements
- Python 3.10+
- pip
- (Optional) Git, if you’re cloning or contributing

### Install & run

```bash
# 1) clone or open your local folder
git clone https://github.com/<you>/slab-nesting.git
cd slab-nesting

# 2) (recommended) virtual env
python -m venv .venv
# Windows PowerShell:
. .\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

# 3) dependencies
pip install -r requirements.txt

# 4) run
streamlit run app.py
# Or:
# python -m streamlit run app.py
