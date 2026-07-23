# DustComp

A code for infrared spectral decomposition of dust spectral features. 

## Instructions

1. Create your own setup_...py file, or adapt the included setup_minds.py for your needs.
2. In the setup file, specify the following:
    - fitter options
    - input data
    - opacity files
3. In run_DustComp.py, set the SETUP_FILE variable to the path of your setup file.
4. Run the code: python3 run_DustComp.py

The package contains example data and results from the "MINDS survey of silicates in T Tauri disks" (Varga et al. 2026, https://ui.adsabs.harvard.edu/abs/2026A%26A...711A.125V/abstract)

## Requirements

### Standard Library (Built-in — No installation required)
* `dataclasses`, `importlib`, `multiprocessing`, `os`, `pathlib`, `pickle`, `time`, `warnings`

### Common Data Science Packages
Likely already in your environment, but install if missing:
```bash
pip install numpy matplotlib scipy
```

### Specialized Dependencies (Must be installed)
These are specific to this project and likely need installation:
* **`numba`**: JIT compiler for high-performance execution
* **`corner`**: For visualizing multi-dimensional posterior distributions
* **`dynesty`**: Dynamic Nested Sampling library

```bash
pip install numba corner dynesty
```


## Dependencies

| Module | Category | Notes / Installation |
| :--- | :--- | :--- |
| `dataclasses`, `importlib`, `multiprocessing`, `os`, `pathlib`, `pickle`, `time`, `warnings` | **Standard Library** | Included with Python |
| `numpy`, `matplotlib`, `scipy` | **Core Data Science** | Common packages (`pip install numpy matplotlib scipy`) |
| `numba` | **Performance** | JIT compiler (`pip install numba`) |
| `corner` | **Visualization** | Diagnostic corner plots (`pip install corner`) |
| `dynesty` | **Sampling** | Nested sampling (`pip install dynesty`) |

### One-line Installation
```bash
pip install numpy matplotlib scipy numba corner dynesty

