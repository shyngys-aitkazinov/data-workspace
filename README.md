# JULIUS BÄR
## Onboard Quest - Improve Client Onboarding Efficiency in Private Banking using Machine Learning and Gamification
In private banking, the onboarding process requires verifying client information against a set of regulatory rules to ensure compliance. This manual process is often time-consuming and error-prone, leading to delays and a poor customer experience. Inconsistencies in documentation represent another major challenge. Onboarding can involve analysing 200-300 pages of information and contracts, where discrepancies are not only common but also significantly impact efficiency. Document and data analysis may seem mundane, but the pain today is substantial.

---

## Project Structure

```bash
├── LICENSE
├── README.md                        # You're reading this!
├── data
│   └── ...
├── main.py                          # Main script or entry point
├── notebooks
│   └── test_prediction.ipynb
├── pyproject.toml                   # Project 
└── uv.lock                          # uv-specific lock file
```
- **`notebooks/`**: Jupyter notebooks for exploration and prototyping.
- **`main.py`**: Example script or possible CLI entry point.
- **`pyproject.toml`**: Defines project requirements, build system, and metadata.

## Prerequisites

- **Python 3.10** (Make sure you have Python 3.10.x installed)
- **uv** (a fast Python package manager) or **pip** (standard Python installer)

### Installing uv

**macOS/Linux**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows** (Powershell):
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

---

## Installation

### Option A: Using `uv` (Recommended)

1. **Clone the repository**:
   ```bash
   git clone git@github.com:shyngys-aitkazinov/data-workspace.git
   cd data-workspace
   git checkout feature/julius_baer
   ```

2. **Create and activate a virtual environment**:
   ```bash
   uv venv --python 3.10
   source .venv/bin/activate
   ```
   *(On Windows, use `./.venv/Scripts/activate`)*

3. **Install the project**:
   ```bash
   uv sync
   ```
   This installs all required dependencies from `pyproject.toml`.

4. **(Optional) Install dev dependencies**:
   ```bash
   uv sync --all-extras
   ```

5. **(Optional) Editable Install** (like `pip install -e .`):
   ```bash
   uv pip install -e .
   ```
   Allows immediate reflection of code changes.

6. **Install specific  pip dependencies**:
   ```bash
   uv pip install 'transformers[torch]'
   ```

### Option B: Using standard pip

1. **Clone the repository**:
   ```bash
   git clone git@github.com:shyngys-aitkazinov/data-workspace.git
   cd data-workspace
   ```
2. **Create and activate a Python 3.10 virtual environment**:
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   or
   ```bash
   pip install -e .
   ```
### Check your installation
```bash
python -c "import torch; print('Package loaded, OK!')"
```
---

## Usage

After installing, run:
```bash
python main.py
```

---

## Development

1. **Install dev dependencies** (if you haven’t already):
   ```bash
   uv sync --all-extras
   ```
2. **Lint and format**:
   ```bash
   ruff check . --fix
   ```
3. **Type checking**:
   ```bash
   mypy .
   ```
4. ** VSCode**: Download ruff and mypy extensions for linting and type checking.
---

## Troubleshooting

- **Multiple top-level packages**: If you see an error about multiple packages discovered (e.g., `data`, `notebooks`, etc.), the `src` layout ensures that only `src/your_package` is treated as a package.
- **Missing library stubs**: If MyPy complains, add `ignore_missing_imports = true` under `[tool.mypy]` in `pyproject.toml`. Aternatively, add a `# type: ignore` comment to the line.
  
- **Git-based installs**: If `uv` fails for certain Git-based sources, manually install with:
  ```bash
  pip install git+https://github.com/some_project/some_repo.git
  ```
  or try
  ```bash
  uv pip install some_package
  ```

---

## License
MIT License. See `LICENSE` for details.


