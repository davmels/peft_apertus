#!/usr/bin/env python3
"""
Generate all analysis plots for the README.

This script generates the 6 core plots:
1. Figure 2 style LR sweep (combined 1x2)
2. 8B training dynamics
3. 70B training dynamics
4. Swiss Judgment accuracy vs rank
5. 8B catastrophic forgetting investigation
6. 70B catastrophic forgetting investigation

Usage:
    python 03_generate_plots.py
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis import Config, DataLoader
from analysis.plotting import PlotGenerator


def generate_figure2_combined(config, data_loader):
    """Generate combined 1x2 Figure 2 style plot."""
    print("Generating Figure 2 style combined (1x2)... ", end='')
    
    metadata = data_loader.load_all_runs_metadata()
    df = metadata[metadata['state'] == 'finished']
    df = df[(df['final_loss'].notna()) & (df['final_loss'] < 2.0)]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    cmap = plt.cm.viridis
    ranks_to_plot = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    
    for idx, model in enumerate(['8B', '70B']):
        ax = axes[idx]
        model_data = df[df['model_size'] == model]
        
        lora_data = model_data[model_data['method'] == 'lora']
        
        if model == '70B':
            lora_data = lora_data[~((lora_data['lora_rank'] == 1) & (lora_data['learning_rate'] == 1e-3))]
        
        available_ranks = sorted([r for r in ranks_to_plot if r in lora_data['lora_rank'].unique()])
        n_ranks = len(available_ranks)
        
        for i, rank in enumerate(available_ranks):
            rank_df = lora_data[lora_data['lora_rank'] == rank].sort_values('learning_rate')
            
            if len(rank_df) < 2:
                continue
            
            lrs = rank_df['learning_rate'].values
            losses = rank_df['final_loss'].values
            color = cmap(i / max(n_ranks - 1, 1))
            
            ax.plot(lrs, losses, marker='o', markersize=8, linewidth=2,
                   label=f'r={rank}', color=color, alpha=0.8)
        
        full_data = model_data[model_data['method'] == 'full']
        if len(full_data) > 0:
            full_sorted = full_data.sort_values('learning_rate')
            ax.plot(full_sorted['learning_rate'], full_sorted['final_loss'],
                   marker='s', markersize=10, linewidth=3, linestyle='-',
                   label='Full FT', color='orange', alpha=0.9, zorder=10)
        
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate', fontweight='bold', fontsize=13)
        ax.set_ylabel('Final Training Loss', fontweight='bold', fontsize=13)
        ax.set_title(f'{model} Model', fontweight='bold', fontsize=15)
        ax.legend(loc='best', fontsize=9, ncol=2, framealpha=0.95)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Learning Rate vs Final Loss (Figure 2 Style)', 
                fontweight='bold', fontsize=16, y=1.00)
    plt.tight_layout()
    
    output_path = config.plots_dir / 'figure2_style_combined.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ {output_path.name}")
    return output_path


def generate_swiss_judgment_plot(config, data_loader):
    """Generate Swiss Judgment accuracy vs rank plot."""
    print("Generating Swiss Judgment accuracy... ", end='')
    
    df = pd.read_csv(config.data_dir / 'swiss_judgment_eval_results.csv')
    df_filtered = df[~((df['model_size'] == '70B') & (df['lr'] == 1e-3) & (df['rank'] == 1))]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors_lr = {1e-5: '#2E86AB', 1e-4: '#A23B72', 1e-3: '#F18F01'}
    markers = {1e-5: 'o', 1e-4: 's', 1e-3: '^'}
    
    for idx, model in enumerate(['8B', '70B']):
        ax = axes[idx]
        data = df_filtered[df_filtered['model_size'] == model]
        
        for lr in sorted(data['lr'].unique()):
            lr_data = data[data['lr'] == lr].sort_values('rank')
            ax.plot(np.log2(lr_data['rank']), lr_data['eval_accuracy'] * 100, 
                   marker=markers[lr], markersize=10, linewidth=2.5,
                   label=f'LR={lr:.0e}', color=colors_lr[lr], alpha=0.8)
        
        ranks = sorted(data['rank'].unique())
        ax.set_xticks(np.log2(ranks))
        ax.set_xticklabels(ranks)
        ax.set_xlabel('LoRA Rank', fontweight='bold', fontsize=13)
        ax.set_ylabel('Swiss Judgment Accuracy (%)', fontweight='bold', fontsize=13)
        ax.set_title(f'{model} Model', fontweight='bold', fontsize=15)
        ax.legend(loc='best', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([70, 90])
    
    fig.text(0.75, -0.02, '*70B excludes rank=1 at LR=1e-3 (diverged)', 
            ha='center', fontsize=10, style='italic')
    
    plt.suptitle('Swiss Judgment Task Performance vs LoRA Rank', 
                fontweight='bold', fontsize=16, y=1.00)
    plt.tight_layout()
    
    output_path = config.plots_dir / 'swiss_judgment_accuracy_vs_rank.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ {output_path.name}")
    return output_path


def generate_investigation_plots(config):
    """Generate catastrophic forgetting investigation plots."""
    df_eval = pd.read_csv(config.data_dir / 'lexam_eval_results.csv')
    df_meta = pd.read_csv(config.data_dir / 'all_runs_parsed.csv')
    
    for model in ['8B', '70B']:
        print(f"Generating {model} investigation plot... ", end='')
        
        # Parse eval data
        eval_data = []
        # Match both patterns: grid_8B_lora_lr and grid_8B_lora_checkpoint
        mask = (df_eval['name'].str.contains(f'grid_{model}_lora', regex=False, na=False) | 
                df_eval['name'].str.contains(f'lr.*lora_{model}', regex=True, na=False))
        for _, run in df_eval[mask].iterrows():
            for col in df_eval.columns:
                if 'mcq_4_choices/accuracy' in col and model in col:
                    val = run[col]
                    if pd.notna(val):
                        parts = col.split('/')[0].split('_')
                        lr_str = parts[0].replace('lr', '')
                        rank_str = parts[1].replace('r', '')
                        
                        eval_data.append({
                            'lr': float(lr_str),
                            'rank': int(rank_str),
                            'lexam_acc': val
                        })
        
        df_eval_parsed = pd.DataFrame(eval_data)
        
        if len(df_eval_parsed) == 0:
            print(f"✗ No eval data for {model}")
            continue
        
        # Merge with training loss
        lora_model = df_meta[(df_meta['model_size'] == model) & (df_meta['method'] == 'lora')]
        lora_model_agg = lora_model.groupby(['learning_rate', 'lora_rank']).agg({
            'final_loss': 'mean'
        }).reset_index()
        
        # Rename columns in eval data to match
        df_eval_parsed = df_eval_parsed.rename(columns={'lr': 'learning_rate', 'rank': 'lora_rank'})
        
        df_merged = df_eval_parsed.merge(
            lora_model_agg, 
            on=['learning_rate', 'lora_rank'],
            how='left'
        )
        
        # Filter out diverged point for 70B
        if model == '70B':
            df_merged = df_merged[~((df_merged['lora_rank'] == 1) & (df_merged['learning_rate'] == 1e-3))]
        
        # Create 1x3 plot
        fig = plt.figure(figsize=(18, 5))
        
        colors = {1e-5: '#2E86AB', 1e-4: '#A23B72', 1e-3: '#F18F01'}
        ranks = sorted(df_merged['lora_rank'].unique())
        
        # 1. Scatter: Training Loss vs LEXam Accuracy
        ax1 = plt.subplot(1, 3, 1)
        for lr in df_merged['learning_rate'].unique():
            data = df_merged[df_merged['learning_rate'] == lr]
            ax1.scatter(data['final_loss'], data['lexam_acc'], 
                       label=f'LR={lr:.0e}', s=120, alpha=0.7, color=colors.get(lr))
        
        ax1.axhline(0.331 if model == '70B' else 0.271, color='green', linestyle='--', 
                   linewidth=2.5, label='Baseline', alpha=0.7)
        ax1.set_xlabel('Training Loss (Lower is Better)', fontweight='bold', fontsize=12)
        ax1.set_ylabel('LEXam Accuracy (Higher is Better)', fontweight='bold', fontsize=12)
        ax1.set_title('Training Loss vs LEXam Accuracy', fontweight='bold', fontsize=14, pad=12)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. Line plot: LEXam Accuracy by Rank
        ax2 = plt.subplot(1, 3, 2)
        for lr in sorted(df_merged['learning_rate'].unique()):
            data = df_merged[df_merged['learning_rate'] == lr].sort_values('lora_rank')
            ax2.plot(range(len(data)), data['lexam_acc'], 'o-', 
                    label=f'LR={lr:.0e}', linewidth=2.5, markersize=9, color=colors.get(lr))
        
        ax2.axhline(0.331 if model == '70B' else 0.271, color='green', linestyle='--', 
                   linewidth=2.5, label='Baseline', alpha=0.7)
        ax2.set_xticks(range(len(ranks)))
        ax2.set_xticklabels(ranks, rotation=45)
        ax2.set_xlabel('LoRA Rank', fontweight='bold', fontsize=12)
        ax2.set_ylabel('LEXam Accuracy', fontweight='bold', fontsize=12)
        ax2.set_title('LEXam Accuracy vs Rank', fontweight='bold', fontsize=14, pad=12)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. Line plot: Training Loss by Rank
        ax3 = plt.subplot(1, 3, 3)
        for lr in sorted(df_merged['learning_rate'].unique()):
            data = df_merged[df_merged['learning_rate'] == lr].sort_values('lora_rank')
            ax3.plot(range(len(data)), data['final_loss'], 'o-', 
                    label=f'LR={lr:.0e}', linewidth=2.5, markersize=9, color=colors.get(lr))
        
        ax3.set_xticks(range(len(ranks)))
        ax3.set_xticklabels(ranks, rotation=45)
        ax3.set_xlabel('LoRA Rank', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Training Loss', fontweight='bold', fontsize=12)
        ax3.set_title('Training Loss vs Rank', fontweight='bold', fontsize=14, pad=12)
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        title_suffix = '*' if model == '70B' else ''
        plt.suptitle(f'Investigation: Training Loss vs LEXam Accuracy ({model} LoRA){title_suffix}', 
                    fontweight='bold', fontsize=16, y=1.02)
        
        if model == '70B':
            fig.text(0.5, -0.02, '*Excludes rank=1 at LR=1e-3 (diverged with loss=9.99)', 
                    ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        
        output_path = config.plots_dir / f'alignment_investigation_{model}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ {output_path.name}")


def main():
    """Generate all plots."""
    
    print("="*80)
    print("GENERATING ALL PLOTS FOR README")
    print("="*80)
    
    # Initialize
    config = Config()
    data_loader = DataLoader(config)
    plotter = PlotGenerator(config, data_loader)
    
    generated_plots = []
    
    # 1. Figure 2 style combined
    print("\n1. Figure 2 Style LR Sweep (Combined 1x2)")
    path = generate_figure2_combined(config, data_loader)
    generated_plots.append(path)
    
    # 2-3. Training dynamics
    print("\n2-3. Training Dynamics")
    for model in ['8B', '70B']:
        print(f"Generating {model} learning curves... ", end='')
        # Use paneled version for 70B (includes zoom window)
        if model == '70B':
            path = plotter.plot_all_learning_curves_paneled(model)
        else:
            path = plotter.plot_all_learning_curves(model)
        print(f"✓ {path.name}")
        generated_plots.append(path)
    
    # 4. Swiss Judgment accuracy
    print("\n4. Swiss Judgment Task Performance")
    path = generate_swiss_judgment_plot(config, data_loader)
    generated_plots.append(path)
    
    # 5-6. Investigation plots
    print("\n5-6. Catastrophic Forgetting Investigation")
    generate_investigation_plots(config)
    generated_plots.append(config.plots_dir / 'alignment_investigation_8B.png')
    generated_plots.append(config.plots_dir / 'alignment_investigation_70B.png')
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✅ PLOT GENERATION COMPLETE")
    print(f"{'='*80}")
    
    print(f"\nGenerated {len(generated_plots)} plots:")
    for i, plot in enumerate(generated_plots, 1):
        print(f"  {i}. {plot.name}")
    
    print(f"\nLocation: {config.plots_dir}")
    print(f"\nAll plots ready for README!")


if __name__ == '__main__':
    main()
