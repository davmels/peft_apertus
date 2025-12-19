#!/usr/bin/env python3
"""
Generate all analysis plots.

This script generates learning curves, LR sweeps, heatmaps,
and model comparisons from the downloaded training data.

Usage:
    python 03_generate_plots.py [--model {8B,70B,all}]
"""
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis import Config, DataLoader
from analysis.plotting import PlotGenerator


def main():
    """Generate all analysis plots."""
    
    parser = argparse.ArgumentParser(description='Generate analysis plots')
    parser.add_argument(
        '--model',
        choices=['8B', '70B', 'all'],
        default='all',
        help='Which model size to plot (default: all)'
    )
    args = parser.parse_args()
    
    print("="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    # Initialize
    config = Config()
    data_loader = DataLoader(config)
    plotter = PlotGenerator(config, data_loader)
    
    # Determine which models to plot
    if args.model == 'all':
        models = config.model_sizes
    else:
        models = [args.model]
    
    print(f"\n📊 Generating plots for: {', '.join(models)}")
    
    generated_plots = []
    
    # 1. Comprehensive learning curves (all LRs in one plot)
    print(f"\n{'='*80}")
    print(f"1. COMPREHENSIVE LEARNING CURVES (All LRs)")
    print(f"{'='*80}\n")
    
    for model in models:
        try:
            print(f"Generating: {model} all LRs... ", end='')
            
            # Use paneled version for 70B, single plot for 8B
            if model == '70B':
                path = plotter.plot_all_learning_curves_paneled(model)
            else:
                path = plotter.plot_all_learning_curves(model)
            
            print(f"✓ {path.name}")
            generated_plots.append(path)
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # 2. LR sweep plots (Figure 2 style)
    print(f"\n{'='*80}")
    print(f"2. LR SWEEP PLOTS (Figure 2 Style)")
    print(f"{'='*80}\n")
    
    for model in models:
        try:
            print(f"Generating: {model} LR sweep... ", end='')
            path = plotter.plot_lr_sweep(model)
            print(f"✓ {path.name}")
            generated_plots.append(path)
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✅ PLOT GENERATION COMPLETE")
    print(f"{'='*80}")
    
    print(f"\nGenerated {len(generated_plots)} plots:")
    print(f"  Location: {config.plots_dir}")
    
    # Organize by type
    learning_curves = [p for p in generated_plots if 'learning_curves' in p.name]
    lr_sweeps = [p for p in generated_plots if 'figure2_style' in p.name]
    
    print(f"\n  - Learning curves: {len(learning_curves)}")
    print(f"  - LR sweeps: {len(lr_sweeps)}")
    
    print(f"\nNext step: Review plots or run script 04_generate_report.py")


if __name__ == '__main__':
    main()

