# Analysis Scripts

Executable pipeline for LoRA hyperparameter analysis.

## Pipeline

```
01_fetch_wandb_data.py → 02_download_history.py → 03_generate_plots.py
```

---

## Scripts

### 1. `01_fetch_wandb_data.py`
**Fetch experiment metadata from WandB**

```bash
python scripts/01_fetch_wandb_data.py
```

**Outputs:**
- `data/runs_overview.csv` - Basic run information
- `data/all_runs_parsed.csv` - Parsed hyperparameters
- `data/metadata.json` - Summary statistics

**Prerequisites:** WandB authentication (`wandb login` or `WANDB_API_KEY`)

---

### 2. `02_download_history.py`
**Download complete training histories**

```bash
python scripts/02_download_history.py [--overwrite]
```

**Options:**
- `--overwrite` - Re-download existing files

**Outputs:**
- `data/history/*.csv` - Training histories (52 runs)
- `data/download_inventory.json` - Download statistics
- Updates `data/all_runs_parsed.csv` with final losses

**Prerequisites:** Must run `01_fetch_wandb_data.py` first

---

### 3. `03_generate_plots.py`
**Generate all analysis plots**

```bash
python scripts/03_generate_plots.py [--model {8B,70B,all}]
```

**Options:**
- `--model` - Which model size to plot (default: all)

**Outputs:**
- `plots/all_learning_curves_8B.png` - Complete training dynamics (8B)
- `plots/all_learning_curves_70B.png` - Complete training dynamics (70B)
- `plots/figure2_style_8B.png` - LR sweep (8B)
- `plots/figure2_style_70B.png` - LR sweep (70B)

**Prerequisites:** Must run `02_download_history.py` first

---

## Quick Start

```bash
# Complete pipeline
python scripts/01_fetch_wandb_data.py
python scripts/02_download_history.py
python scripts/03_generate_plots.py
```

---

## Configuration

All configuration is centralized in `analysis/config.py`:

- **WandB project:** `lsaie-peft-apertus/swiss_judgment_prediction`
- **Learning rates:** `[1e-5, 1e-4, 1e-3]`
- **LoRA ranks:** `[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]`
- **Model sizes:** `['8B', '70B']`

---

## Troubleshooting

### WandB Authentication Error
```
Error: No API key configured
```
**Solution:** Run `wandb login` or set `WANDB_API_KEY` environment variable

### Missing Metadata Error
```
Error: Run metadata not found
```
**Solution:** Run `01_fetch_wandb_data.py` first

### Empty Plots
**Solution:** Check `data/all_runs_parsed.csv` for available runs

---

## Dependencies

All scripts use the shared `analysis` package:

```python
# Core
pandas>=1.5.0
numpy>=1.23.0

# Plotting
matplotlib>=3.6.0

# WandB
wandb>=0.15.0

# Utilities
tqdm>=4.65.0
```

---

*For detailed package documentation, see `../analysis/` modules.*
