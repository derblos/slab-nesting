# Installation Guide for Slab Nesting Tool

This guide will walk you through setting up the Slab Nesting application step-by-step.

---

## Prerequisites

Before you begin, ensure you have:
- **Python 3.10 or higher** (3.10, 3.11, or 3.12 recommended)
- **Internet connection** (for downloading dependencies)
- **Command line access** (PowerShell on Windows, Terminal on macOS/Linux)

---

## Choose Your Installation Method

### 🟢 Recommended: Conda (Best for Windows)

Conda is a package manager that handles dependencies reliably, especially on Windows.

#### Step 1: Install Miniconda

1. Download Miniconda for your operating system:
   - **Windows**: https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
   - **macOS**: https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
   - **Linux**: https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

2. Run the installer:
   - **Windows**: Double-click the downloaded `.exe` file and follow the prompts
   - **macOS/Linux**: Open Terminal and run:
     ```bash
     bash Miniconda3-latest-*.sh
     ```

3. **Important for Windows users**: When asked "Add Miniconda3 to PATH", you can choose either option:
   - ✅ **Yes**: Allows you to use conda from any terminal
   - ⚠️ **No**: You'll need to use "Miniconda Prompt" from the Start menu

#### Step 2: Download the Project

1. Open a terminal:
   - **Windows**: Search for "Miniconda Prompt" in the Start menu (or PowerShell if you added to PATH)
   - **macOS/Linux**: Open your Terminal application

2. Navigate to where you want to install the project:
   ```bash
   cd C:\Users\YourUsername\Documents\GitHub  # Windows
   cd ~/Documents/GitHub                      # macOS/Linux
   ```

3. Clone the repository (if using Git):
   ```bash
   git clone https://github.com/your-username/slab-nesting.git
   cd slab-nesting
   ```

   Or download and extract the ZIP file, then navigate to it:
   ```bash
   cd slab-nesting
   ```

#### Step 3: Create the Conda Environment

1. Create the environment from the provided file:
   ```bash
   conda env create -f environment.yml
   ```

   This will:
   - Create a new environment named "slab-nesting"
   - Install Python 3.10+
   - Install all required packages (plotly, shapely, streamlit, etc.)

2. Wait for the installation to complete (may take 2-5 minutes)

#### Step 4: Activate the Environment

Every time you want to use this application, you need to activate the environment:

```bash
conda activate slab-nesting
```

You should see `(slab-nesting)` appear before your command prompt.

#### Step 5: Run the Application

```bash
streamlit run app.py
```

The application should open automatically in your default web browser at `http://localhost:8501`

---

### 🔵 Alternative: pip with Virtual Environment

If you prefer not to use Conda or already have Python installed:

#### Step 1: Verify Python Installation

Open a terminal and check your Python version:

```bash
python --version
# or
python3 --version
```

Ensure it's Python 3.10 or higher.

#### Step 2: Download the Project

Follow the same instructions as in "Step 2" above.

#### Step 3: Create a Virtual Environment

```bash
# Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` appear before your command prompt.

#### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages.

#### Step 5: Run the Application

```bash
streamlit run app.py
```

---

## Troubleshooting Common Issues

### Issue: "conda: command not found"

**Solution**:
- If on Windows: Use "Miniconda Prompt" from Start menu instead of regular PowerShell
- If you want to use regular PowerShell: Reinstall Miniconda and check "Add to PATH"
- Alternatively: Use the pip installation method

### Issue: "No module named 'plotly'" (or other modules)

**Solution**:
```bash
# For conda users:
conda activate slab-nesting
conda env update -f environment.yml --prune

# For pip users:
pip install -r requirements.txt --upgrade
```

### Issue: "Activate.ps1 cannot be loaded" (Windows PowerShell)

**Solution**: PowerShell execution policy issue. Run this once:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating again.

### Issue: Port 8501 is already in use

**Solution**: Stop the running Streamlit instance or use a different port:
```bash
streamlit run app.py --server.port 8502
```

### Issue: Application is very slow

**Possible causes**:
- Very large canvas sizes in settings
- Nesting hundreds of parts at once
- Insufficient RAM (<4GB)

**Solutions**:
- Reduce canvas dimensions in sidebar
- Nest smaller batches of parts
- Close other memory-intensive applications

---

## Updating the Application

When new versions are released:

### For Conda Users:
```bash
conda activate slab-nesting
git pull  # or download new version
conda env update -f environment.yml --prune
```

### For pip Users:
```bash
source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows
git pull  # or download new version
pip install -r requirements.txt --upgrade
```

---

## Daily Usage Workflow

### For Conda Users (Windows - Easiest):

1. Open **Miniconda Prompt** from Start menu
2. Navigate to project:
   ```bash
   cd C:\Users\YourUsername\Documents\GitHub\slab-nesting
   ```
3. Activate environment:
   ```bash
   conda activate slab-nesting
   ```
4. Run application:
   ```bash
   streamlit run app.py
   ```

### For pip Users:

1. Open Terminal/PowerShell
2. Navigate to project directory
3. Activate virtual environment:
   ```bash
   # Windows:
   .\.venv\Scripts\Activate.ps1

   # macOS/Linux:
   source .venv/bin/activate
   ```
4. Run application:
   ```bash
   streamlit run app.py
   ```

---

## Uninstalling

### For Conda Users:
```bash
conda env remove -n slab-nesting
```

### For pip Users:
Simply delete the project folder and the `.venv` directory inside it.

---

## Getting Help

If you encounter issues not covered here:

1. Check the error message carefully - it often tells you what's wrong
2. Ensure you've activated the environment (`conda activate slab-nesting`)
3. Try reinstalling dependencies
4. Check GitHub Issues for similar problems
5. Create a new issue with:
   - Your operating system and Python version
   - The complete error message
   - Steps to reproduce the problem

---

## Next Steps

Once installed successfully:
- Read the main README.md for feature overview
- Explore the application interface
- Try the sample workflows in the documentation
- Adjust settings in the sidebar to match your needs

Happy nesting! 🔲
