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
- Python 3.10+ (up to 3.12 recommended)
- **Conda** (Miniconda/Anaconda) or **pip**
- Git (optional, for cloning)

---

### Installation

#### **Option 1: Using Conda (Recommended for Windows)**

This is the **easiest and most reliable** method, especially on Windows.

1. **Install Miniconda** (if not already installed):
   - Download from: https://docs.conda.io/en/latest/miniconda.html
   - Run the installer and follow the prompts

2. **Clone or download this repository**:
   ```bash
   git clone https://github.com/<your-username>/slab-nesting.git
   cd slab-nesting
   ```

3. **Create the conda environment**:
   ```bash
   conda env create -f environment.yml
   ```

4. **Activate the environment**:
   ```bash
   conda activate slab-nesting
   ```

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

#### **Option 2: Using pip with Virtual Environment**

1. **Clone or download this repository**:
   ```bash
   git clone https://github.com/<your-username>/slab-nesting.git
   cd slab-nesting
   ```

2. **Create a virtual environment**:
   ```bash
   # Windows PowerShell:
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # macOS/Linux:
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

---

### Important Notes for Conda Users

**Every time you work on this project**, you need to activate the conda environment first:

#### **Method 1: Using Miniconda Prompt (Easiest)**
1. Open **Miniconda3 Prompt** from Windows Start menu
2. Navigate to your project:
   ```bash
   cd C:\Users\<YourUsername>\Documents\GitHub\slab-nesting
   ```
3. Activate the environment:
   ```bash
   conda activate slab-nesting
   ```
4. Run Streamlit:
   ```bash
   streamlit run app.py
   ```

#### **Method 2: Using Regular PowerShell/Terminal**
If you've run `conda init powershell` (one-time setup), you can use your regular terminal:
1. Open PowerShell
2. Activate the environment:
   ```bash
   conda activate slab-nesting
   ```
3. Navigate to your project and run Streamlit

---

### Updating Dependencies

If dependencies are updated in the future:

**For conda users**:
```bash
conda env update -f environment.yml --prune
```

**For pip users**:
```bash
pip install -r requirements.txt --upgrade
```

---

### Troubleshooting

#### ❌ "No module named 'plotly'" (or similar errors)
- **Solution**: Make sure you've installed all dependencies:
  - Conda: `conda env update -f environment.yml --prune`
  - pip: `pip install -r requirements.txt`

#### ❌ "conda: command not found"
- **Solution**: Install Miniconda/Anaconda first, or use the pip method

#### ❌ Application won't start
- **Solution**: Ensure you've activated the environment (`conda activate slab-nesting`)
- Check Python version: `python --version` (should be 3.10+)

