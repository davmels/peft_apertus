"""
Plotting utilities for visualizing LoRA experiments.
"""
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config
from .data_loader import DataLoader


class PlotGenerator:
    """Generator for analysis plots."""
    
    def __init__(self, config: Config, data_loader: DataLoader):
        """
        Initialize plot generator.
        
        Args:
            config: Configuration object
            data_loader: Data loader instance
        """
        self.config = config
        self.data = data_loader
        
        # Set default plotting style
        plt.style.use(config.plot_style)
        plt.rcParams['figure.figsize'] = config.default_figsize
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['lines.linewidth'] = 2.5
    
    def _save_figure(self, filename: str) -> Path:
        """Save current figure to plots directory."""
        filepath = self.config.plots_dir / filename
        plt.savefig(
            filepath,
            dpi=self.config.figure_dpi,
            bbox_inches='tight',
            facecolor='white'
        )
        plt.close()
        return filepath
    
    def plot_learning_curves(
        self,
        model_size: str,
        learning_rate: float,
        show_full_ft: bool = True
    ) -> Path:
        """
        Plot learning curves for different ranks at a fixed LR.
        
        Args:
            model_size: Model size ('8B' or '70B')
            learning_rate: Learning rate to plot
            show_full_ft: Whether to include Full FT baseline
        
        Returns:
            Path to saved plot
        """
        metadata = self.data.load_all_runs_metadata()
        
        fig, ax = plt.subplots(figsize=self.config.default_figsize)
        
        # Plot LoRA runs
        for rank in self.config.lora_ranks:
            runs = self.data.filter_runs(
                metadata,
                model_size=model_size,
                method='lora',
                learning_rate=learning_rate,
                lora_rank=rank
            )
            
            if len(runs) == 0:
                continue
            
            run_name = runs.iloc[0]['name']
            history = self.data.load_run_history(run_name)
            
            if history is None or 'train/loss' not in history.columns:
                continue
            
            steps = history['_step'].values
            losses = history['train/loss'].values
            
            color = self.config.get_color_for_rank(rank)
            ax.plot(steps, losses, label=f'r={rank}', color=color, alpha=0.8)
        
        # Plot Full FT
        if show_full_ft:
            full_runs = self.data.filter_runs(
                metadata,
                model_size=model_size,
                method='full',
                learning_rate=learning_rate
            )
            
            if len(full_runs) > 0:
                run_name = full_runs.iloc[0]['name']
                history = self.data.load_run_history(run_name)
                
                if history is not None and 'train/loss' in history.columns:
                    steps = history['_step'].values
                    losses = history['train/loss'].values
                    ax.plot(
                        steps, losses,
                        label='Full FT',
                        color=self.config.full_ft_color,
                        linewidth=3,
                        alpha=0.9
                    )
        
        ax.set_xlabel('Training Steps', fontweight='bold')
        ax.set_ylabel('Training Loss', fontweight='bold')
        ax.set_title(
            f'Learning Curves - {model_size} Model (LR={learning_rate:.0e})',
            fontweight='bold',
            pad=20
        )
        ax.legend(loc='best', framealpha=0.95, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        lr_str = f"{learning_rate:.0e}".replace('e-0', 'e-')
        filename = f"learning_curves_{model_size}_lr{lr_str}.png"
        return self._save_figure(filename)
    
    def plot_lr_sweep(
        self,
        model_size: str,
        show_full_ft: bool = True
    ) -> Path:
        """
        Plot LR vs final loss (Figure 2 style from blog).
        
        Args:
            model_size: Model size ('8B' or '70B')
            show_full_ft: Whether to include Full FT baseline
        
        Returns:
            Path to saved plot
        """
        metadata = self.data.load_all_runs_metadata()
        
        # Filter and clean data
        lora_df = self.data.filter_runs(
            metadata,
            model_size=model_size,
            method='lora'
        )
        lora_df = lora_df[
            (lora_df['final_loss'].notna()) &
            (lora_df['final_loss'] < 2.0)  # Filter diverged runs
        ].copy()
        
        full_df = self.data.filter_runs(
            metadata,
            model_size=model_size,
            method='full'
        )
        full_df = full_df[
            (full_df['final_loss'].notna()) &
            (full_df['final_loss'] < 2.0)
        ].copy()
        
        fig, ax = plt.subplots(figsize=self.config.default_figsize)
        
        # Plot LoRA runs
        for rank in self.config.lora_ranks:
            rank_df = lora_df[lora_df['lora_rank'] == rank].sort_values('learning_rate')
            
            if len(rank_df) < 2:
                continue
            
            lrs = rank_df['learning_rate'].values
            losses = rank_df['final_loss'].values
            
            # Remove NaN values
            mask = ~np.isnan(losses)
            lrs = lrs[mask]
            losses = losses[mask]
            
            if len(lrs) >= 2:
                color = self.config.get_color_for_rank(rank)
                ax.plot(
                    lrs, losses, 'o-',
                    label=f'{int(rank)}',
                    color=color,
                    markersize=8,
                    linewidth=2,
                    alpha=0.8
                )
        
        # Plot Full FT
        if show_full_ft and len(full_df) >= 2:
            full_df = full_df.sort_values('learning_rate')
            lrs = full_df['learning_rate'].values
            losses = full_df['final_loss'].values
            mask = ~np.isnan(losses)
            lrs = lrs[mask]
            losses = losses[mask]
            
            if len(lrs) >= 2:
                ax.plot(
                    lrs, losses, 'o-',
                    label='full',
                    color=self.config.full_ft_color,
                    markersize=10,
                    linewidth=3,
                    alpha=0.9
                )
        
        # Styling
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate', fontweight='bold')
        ax.set_ylabel('Final Training Loss', fontweight='bold')
        ax.set_title(
            f'Learning Rate vs Final Loss - {model_size} Model\n'
            f'(Swiss Judgment Prediction)',
            fontweight='bold',
            pad=20
        )
        ax.legend(title='Rank', loc='upper left', framealpha=0.95,
                 title_fontsize=12, fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3, which='both', linestyle='--')
        ax.set_axisbelow(True)
        
        # Set y-axis limits
        all_losses = lora_df['final_loss'].dropna()
        if len(full_df) > 0:
            all_losses = pd.concat([all_losses, full_df['final_loss'].dropna()])
        
        if len(all_losses) > 0:
            data_min = all_losses.min()
            data_max = all_losses.max()
            data_range = data_max - data_min
            y_min = data_min - 0.05 * data_range
            y_max = data_max + 0.05 * data_range
            ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        
        filename = f"figure2_style_{model_size}.png"
        return self._save_figure(filename)
    
    def plot_heatmap(
        self,
        model_size: str,
        metric: str = 'final_loss'
    ) -> Path:
        """
        Plot heatmap of metric vs LR and rank.
        
        Args:
            model_size: Model size ('8B' or '70B')
            metric: Metric to plot ('final_loss', 'min_loss', etc.)
        
        Returns:
            Path to saved plot
        """
        metadata = self.data.load_all_runs_metadata()
        
        lora_df = self.data.filter_runs(
            metadata,
            model_size=model_size,
            method='lora'
        )
        
        # Create pivot table
        pivot = lora_df.pivot_table(
            values=metric,
            index='lora_rank',
            columns='learning_rate',
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(pivot.values, cmap='viridis_r', aspect='auto')
        
        # Set ticks
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels([f"{lr:.0e}" for lr in pivot.columns])
        ax.set_yticklabels([f"{int(r)}" for r in pivot.index])
        
        ax.set_xlabel('Learning Rate', fontweight='bold')
        ax.set_ylabel('LoRA Rank', fontweight='bold')
        ax.set_title(
            f'Heatmap: {metric.replace("_", " ").title()} - {model_size} Model',
            fontweight='bold',
            pad=20
        )
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric.replace('_', ' ').title(), rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        filename = f"lr_rank_heatmap_{model_size}.png"
        return self._save_figure(filename)
    
    def plot_all_learning_curves_paneled(
        self,
        model_size: str,
        show_full_ft: bool = True
    ) -> Path:
        """
        Plot learning curves in a single plot with log y-axis and unified color scheme.
        Uses color gradients (blue/red/green) for different LRs, matching 8B style.
        
        Args:
            model_size: Model size ('8B' or '70B')
            show_full_ft: Whether to include Full FT baselines
        
        Returns:
            Path to saved plot
        """
        metadata = self.data.load_all_runs_metadata()
        
        fig, ax = plt.subplots(figsize=(14, 9))
        
        # Use unified color scheme (4 colors for 4 LRs)
        lr_colormaps = {
            1e-3: plt.cm.Blues,   # Blue for 1e-3
            1e-4: plt.cm.Reds,    # Red for 1e-4
            1e-5: plt.cm.Greens,  # Green for 1e-5
            1e-6: plt.cm.YlOrBr,  # Yellow/Orange for 1e-6
        }
        
        # Representative ranks for 70B (excluding r=1 due to instability at high LR)
        representative_ranks = [16, 64, 512]
        
        # Color intensity range (much more vibrant)
        color_range = (0.4, 0.95)
        
        handles_dict = {}
        
        # Plot LoRA runs grouped by LR
        for lr in sorted(self.config.learning_rates, reverse=True):
            cmap = lr_colormaps.get(lr, plt.cm.Greys)
            
            lr_runs = self.data.filter_runs(
                metadata,
                model_size=model_size,
                method='lora',
                learning_rate=lr
            )
            
            if len(lr_runs) == 0:
                continue
            
            available_ranks = sorted(lr_runs['lora_rank'].dropna().unique())
            plot_ranks = [r for r in representative_ranks if r in available_ranks]
            
            if len(plot_ranks) == 0:
                continue
            
            n_ranks = len(plot_ranks)
            
            # Normalize for color mapping
            norm = plt.Normalize(min(plot_ranks), max(plot_ranks))
            
            lr_str = f"{lr:.0e}".replace('e-0', 'e-')
            handles_dict[lr_str] = []
            
            for i, rank in enumerate(plot_ranks):
                runs = lr_runs[lr_runs['lora_rank'] == rank]
                
                if len(runs) == 0:
                    continue
                
                run_name = runs.iloc[0]['name']
                history = self.data.load_run_history(run_name)
                
                if history is None or 'train/loss' not in history.columns:
                    continue
                
                steps = history['_step'].values
                losses = history['train/loss'].values
                
                # Get color from colormap
                color_intensity = norm(rank)
                color = cmap(color_range[0] + color_intensity * (color_range[1] - color_range[0]))
                
                # Line thickness (thinner)
                linewidth = 1.5 if rank in [64, 512] else 1.0
                
                line, = ax.plot(steps, losses, color=color, linestyle='-',
                               linewidth=linewidth, alpha=0.85, label=f'r={int(rank)}')
                
                handles_dict[lr_str].append((line, f'r={int(rank)}'))
        
        # Plot Full FT baselines with fluorescent colors and dotted lines
        # Full FT uses 10x lower LR than LoRA, so we map colors accordingly
        if show_full_ft:
            # Map LoRA LR to Full FT LR (10x lower) and color
            # Colors match the LR, using fluorescent tones for visibility
            lora_to_fullft_mapping = {
                1e-3: (1e-4, '#404040', ':'),    # LoRA@1e-3 → Full FT@1e-4 (Dark gray, dotted)
                1e-4: (1e-5, '#707070', '-.'),   # LoRA@1e-4 → Full FT@1e-5 (Mid gray, dash-dot)
                1e-5: (1e-6, '#A0A0A0', '--'),   # LoRA@1e-5 → Full FT@1e-6 (Light gray, dashed)
            }
            
            handles_dict['Full FT'] = []
            
            # Plot Full FT at 10x lower LRs with matching colors (reverse order: 1e-6, 1e-5, 1e-4)
            for lora_lr, (ft_lr, ft_color, ft_linestyle) in reversed(list(lora_to_fullft_mapping.items())):
                full_runs = self.data.filter_runs(
                    metadata,
                    model_size=model_size,
                    method='full',
                    learning_rate=ft_lr  # Use 10x lower LR
                )
                
                if len(full_runs) > 0:
                    run_name = full_runs.iloc[0]['name']
                    history = self.data.load_run_history(run_name)
                    
                    if history is not None and 'train/loss' in history.columns:
                        steps = history['_step'].values
                        losses = history['train/loss'].values
                        
                        ft_lr_str = f"{ft_lr:.0e}".replace('e-0', 'e-')
                        
                        line, = ax.plot(steps, losses, color=ft_color, linewidth=1.8,
                                       linestyle=ft_linestyle, alpha=0.9, label=f'Full FT', zorder=10)
                        
                        handles_dict['Full FT'].append((line, f'LR={ft_lr_str}'))
        
        # Formatting with log y-axis
        ax.set_xlabel('Training Steps', fontweight='bold', fontsize=15)
        ax.set_ylabel('Training Loss (log scale)', fontweight='bold', fontsize=15)
        ax.set_title(f'Learning Curves - {model_size} Model (All Learning Rates)',
                    fontweight='bold', fontsize=18, pad=20)
        
        # Set log scale
        ax.set_yscale('log')
        
        # Create 4-column legend with proper color grouping
        # Each column: 1 header + 3 ranks = 4 rows per column
        # Columns: Blue (LR=1e-3), Red (LR=1e-4), Green (LR=1e-5), Full FT
        
        padded_handles = []
        padded_labels = []
        
        color_names = {
            '1e-3': 'Blue',
            '1e-4': 'Red',
            '1e-5': 'Green'
        }
        
        # Build each column (simpler padding logic)
        for lr_str in ['1e-3', '1e-4', '1e-5']:
            if lr_str in handles_dict:
                # Header
                color_name = color_names.get(lr_str, '')
                padded_handles.append(plt.Line2D([0], [0], color='none'))
                padded_labels.append(f'LR={lr_str} ({color_name})')
                
                # Ranks
                num_ranks = 0
                for line, label in handles_dict[lr_str]:
                    padded_handles.append(line)
                    padded_labels.append(label)
                    num_ranks += 1
                
                # Pad to 4 rows total (1 header + 3 ranks)
                while num_ranks < 3:
                    padded_handles.append(plt.Line2D([0], [0], color='none'))
                    padded_labels.append('')
                    num_ranks += 1
        
        # Full FT column
        if 'Full FT' in handles_dict:
            padded_handles.append(plt.Line2D([0], [0], color='none'))
            padded_labels.append('Full FT')
            
            for line, label in handles_dict['Full FT']:
                padded_handles.append(line)
                padded_labels.append(label)
            
            # Pad to 4 rows
            ft_count = len(handles_dict['Full FT']) + 1  # +1 for header
            while ft_count < 4:
                padded_handles.append(plt.Line2D([0], [0], color='none'))
                padded_labels.append('')
                ft_count += 1
        
        ax.legend(
            handles=padded_handles,
            labels=padded_labels,
            loc='upper right',
            framealpha=0.95,
            fontsize=9,
            ncol=4,
            columnspacing=0.7,
            handlelength=1.5
        )
        
        # Add zoom inset for last 100 steps (70B only)
        if model_size == '70B':
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
            
            # Create inset in lower area (larger and lower)
            axins = inset_axes(ax, width="50%", height="40%", loc='center',
                              bbox_to_anchor=(0.25, 0.15, 0.9, 0.9),
                              bbox_transform=ax.transAxes)
            
            # Plot LoRA runs in inset (last 100 steps)
            for lr in sorted(self.config.learning_rates, reverse=True):
                cmap = lr_colormaps.get(lr, plt.cm.Greys)
                
                lr_runs = self.data.filter_runs(
                    metadata,
                    model_size=model_size,
                    method='lora',
                    learning_rate=lr
                )
                
                if len(lr_runs) == 0:
                    continue
                
                available_ranks = sorted(lr_runs['lora_rank'].dropna().unique())
                plot_ranks = [r for r in representative_ranks if r in available_ranks]
                
                if len(plot_ranks) == 0:
                    continue
                
                norm = plt.Normalize(min(plot_ranks), max(plot_ranks))
                
                for i, rank in enumerate(plot_ranks):
                    runs = lr_runs[lr_runs['lora_rank'] == rank]
                    
                    if len(runs) == 0:
                        continue
                    
                    run_name = runs.iloc[0]['name']
                    history = self.data.load_run_history(run_name)
                    
                    if history is None or 'train/loss' not in history.columns:
                        continue
                    
                    steps = history['_step'].values
                    losses = history['train/loss'].values
                    
                    # Filter for last 100 steps
                    if len(steps) <= 100:
                        steps_zoom = steps
                        losses_zoom = losses
                    else:
                        steps_zoom = steps[-100:]
                        losses_zoom = losses[-100:]
                    
                    if len(steps_zoom) == 0:
                        continue
                    
                    color_intensity = norm(rank)
                    color = cmap(color_range[0] + color_intensity * (color_range[1] - color_range[0]))
                    
                    linewidth = 1.0 if rank in [64, 512] else 0.8
                    
                    axins.plot(steps_zoom, losses_zoom, color=color, linestyle='-',
                              linewidth=linewidth, alpha=0.85)
            
            # Plot Full FT in inset (10x lower LRs with matching colors)
            if show_full_ft:
                for lora_lr, (ft_lr, ft_color, ft_linestyle) in lora_to_fullft_mapping.items():
                    full_runs = self.data.filter_runs(
                        metadata,
                        model_size=model_size,
                        method='full',
                        learning_rate=ft_lr  # Use 10x lower LR
                    )
                    
                    if len(full_runs) > 0:
                        run_name = full_runs.iloc[0]['name']
                        history = self.data.load_run_history(run_name)
                        
                        if history is not None and 'train/loss' in history.columns:
                            steps = history['_step'].values
                            losses = history['train/loss'].values
                            
                            # Filter for last 100 steps
                            if len(steps) <= 100:
                                steps_zoom = steps
                                losses_zoom = losses
                            else:
                                steps_zoom = steps[-100:]
                                losses_zoom = losses[-100:]
                            
                            if len(steps_zoom) > 0:
                                axins.plot(steps_zoom, losses_zoom, color=ft_color,
                                          linewidth=1.4, linestyle=ft_linestyle, alpha=0.9, zorder=10)
            
            # Format inset
            axins.set_xlabel('Steps', fontsize=9)
            axins.set_ylabel('Loss', fontsize=9)
            axins.set_title('Final 100 Steps', fontsize=9, pad=5)
            axins.tick_params(labelsize=8)
            axins.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
            axins.ticklabel_format(style='plain', axis='x')
            
            # Mark the zoomed region (inverted connectors)
            mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="0.5", linestyle='--', linewidth=1)
        
        ax.grid(True, alpha=0.25, which='both', linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.ticklabel_format(style='plain', axis='x')
        
        plt.tight_layout()
        
        filename = f"all_learning_curves_{model_size}.png"
        return self._save_figure(filename)
    
    def plot_all_learning_curves(
        self,
        model_size: str,
        show_full_ft: bool = True
    ) -> Path:
        """
        Plot learning curves for all LRs and ranks in a single plot.
        Uses color gradients with log y-axis for better visibility.
        
        Args:
            model_size: Model size ('8B' or '70B')
            show_full_ft: Whether to include Full FT baselines
        
        Returns:
            Path to saved plot
        """
        metadata = self.data.load_all_runs_metadata()
        
        fig, ax = plt.subplots(figsize=(14, 9))
        
        # Use color gradients with vibrant colors (avoid very dark tones)
        # Adjust colormap range to use lighter, more vibrant colors
        lr_colormaps = {
            1e-3: plt.cm.Blues,   # Blues for highest LR
            1e-4: plt.cm.Reds,    # Reds for middle LR (will adjust range)
            1e-5: plt.cm.Greens,  # Greens for lowest LR (will adjust range)
        }
        
        # Representative ranks to show (avoid overcrowding)
        representative_ranks_8b = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        representative_ranks_70b = [16, 64, 512]  # Exclude r=1 for 70B (unstable at LR=1e-3)
        
        rep_ranks = representative_ranks_8b if model_size == '8B' else representative_ranks_70b
        
        # Plot LoRA runs grouped by LR
        handles_dict = {}  # For custom legend ordering
        
        for lr in sorted(self.config.learning_rates, reverse=True):
            cmap = lr_colormaps.get(lr, plt.cm.Greys)
            
            # Get ranks that have data for this LR
            lr_runs = self.data.filter_runs(
                metadata,
                model_size=model_size,
                method='lora',
                learning_rate=lr
            )
            
            if len(lr_runs) == 0:
                continue
            
            available_ranks = sorted(lr_runs['lora_rank'].dropna().unique())
            
            # Filter to representative ranks
            plot_ranks = [r for r in rep_ranks if r in available_ranks]
            
            if len(plot_ranks) == 0:
                continue
            
            n_ranks = len(plot_ranks)
            lr_str = f"{lr:.0e}".replace('e-0', 'e-')
            
            for i, rank in enumerate(plot_ranks):
                runs = lr_runs[lr_runs['lora_rank'] == rank]
                
                if len(runs) == 0:
                    continue
                
                run_name = runs.iloc[0]['name']
                history = self.data.load_run_history(run_name)
                
                if history is None or 'train/loss' not in history.columns:
                    continue
                
                steps = history['_step'].values
                losses = history['train/loss'].values
                
                # Color intensity: lighter → darker as rank increases
                # Use range 0.25-0.85 for vibrant colors (avoid very dark tones)
                # Low ranks (r=1,2,4) lighter, high ranks (r=64,128,256) darker/fainter
                color_intensity = 0.25 + 0.60 * (i / max(n_ranks - 1, 1))
                color = cmap(color_intensity)
                
                # Emphasize key ranks with thicker lines - slimmer
                linewidth = 1.5 if rank in [64, 256, 512] else 1.0
                
                label = f'r={int(rank)}'
                
                line, = ax.plot(steps, losses, color=color, linestyle='-',
                               linewidth=linewidth, alpha=0.85, label=label)
                
                # Group by LR for legend
                if lr_str not in handles_dict:
                    handles_dict[lr_str] = []
                handles_dict[lr_str].append((line, label))
        
        # Plot Full FT baselines with fluorescent colors and dotted lines
        # Full FT uses 10x lower LR than LoRA, so we map colors accordingly
        if show_full_ft:
            # Map LoRA LR to Full FT LR (10x lower) and color
            # Colors match the LR, using fluorescent tones for visibility
            lora_to_fullft_mapping = {
                1e-3: (1e-4, '#404040', ':'),    # LoRA@1e-3 → Full FT@1e-4 (Dark gray, dotted)
                1e-4: (1e-5, '#707070', '-.'),   # LoRA@1e-4 → Full FT@1e-5 (Mid gray, dash-dot)
                1e-5: (1e-6, '#A0A0A0', '--'),   # LoRA@1e-5 → Full FT@1e-6 (Light gray, dashed)
            }
            
            # Plot Full FT at 10x lower LRs with matching colors (reverse order: 1e-6, 1e-5, 1e-4)
            for lora_lr, (ft_lr, ft_color, ft_linestyle) in reversed(list(lora_to_fullft_mapping.items())):
                full_runs = self.data.filter_runs(
                    metadata,
                    model_size=model_size,
                    method='full',
                    learning_rate=ft_lr  # Use the 10x lower LR
                )
                
                if len(full_runs) > 0:
                    run_name = full_runs.iloc[0]['name']
                    history = self.data.load_run_history(run_name)
                    
                    if history is not None and 'train/loss' in history.columns:
                        steps = history['_step'].values
                        losses = history['train/loss'].values
                        
                        lora_lr_str = f"{lora_lr:.0e}".replace('e-0', 'e-')
                        ft_lr_str = f"{ft_lr:.0e}".replace('e-0', 'e-')
                        
                        line, = ax.plot(
                            steps, losses,
                            color=ft_color,
                            linewidth=1.8,
                            linestyle=ft_linestyle,  # Use specific linestyle
                            alpha=0.9,
                            label=f'Full FT (LR={ft_lr_str})',
                            zorder=10  # Draw on top
                        )
                        
                        # Store Full FT separately for end of legend
                        if 'Full FT' not in handles_dict:
                            handles_dict['Full FT'] = []
                        handles_dict['Full FT'].append((line, f'LR={ft_lr_str}'))
        
        # Formatting with linear scale (8B) or log scale (70B)
        ax.set_xlabel('Training Steps', fontweight='bold', fontsize=15)
        
        # Use linear scale for 8B, log scale for 70B
        if model_size == '8B':
            ax.set_ylabel('Training Loss', fontweight='bold', fontsize=15)
        else:
            ax.set_ylabel('Training Loss (log scale)', fontweight='bold', fontsize=15)
            ax.set_yscale('log')
        
        ax.set_title(
            f'Learning Curves - {model_size} Model (All Learning Rates)',
            fontweight='bold',
            fontsize=18,
            pad=20
        )
        
        # Format axes
        ax.ticklabel_format(style='plain', axis='x')
        
        # Custom legend by LR groups
        from matplotlib.lines import Line2D
        legend_elements = []
        
        # Add LoRA runs by LR (excluding 'Full FT' key)
        for lr_str in sorted([k for k in handles_dict.keys() if k != 'Full FT'], reverse=True):
            try:
                lr_val = float(lr_str.replace('e-', 'e-'))
                color_name = {1e-3: 'Blue', 1e-4: 'Red', 1e-5: 'Green'}.get(lr_val, 'Gray')
            except:
                continue
            
            # Add LR header
            legend_elements.append(
                Line2D([0], [0], color='none', label=f'LR={lr_str} ({color_name})')
            )
            
            # Add lines for this LR
            for line, label in handles_dict[lr_str]:
                legend_elements.append(line)
            
            # Add padding to align columns (need 10 items per column for 4-column layout)
            # Each LR has 1 header + 9 ranks = 10 items, perfect for 4 columns
        
        # Add Full FT at the end with padding if needed
        if 'Full FT' in handles_dict:
            legend_elements.append(
                Line2D([0], [0], color='none', label='Full FT')
            )
            for line, label in handles_dict['Full FT']:
                legend_elements.append(line)
            
            # Pad the Full FT column to match height (10 items per column)
            # Full FT has 1 header + 3 LRs = 4 items, need 6 more blanks
            for _ in range(6):
                legend_elements.append(Line2D([0], [0], color='none', label=''))
        
        # Use 4 columns for both 8B and 70B (better organization by color/LR)
        ncols = 4
        
        ax.legend(
            handles=legend_elements,
            loc='upper right',
            framealpha=0.95,
            fontsize=8.5,
            ncol=ncols,
            columnspacing=0.7,
            handlelength=1.5
        )
        
        ax.grid(True, alpha=0.25, which='both', linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Add zoom inset for 8B to show convergence details (step 400+)
        if model_size == '8B':
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
            
            # Create inset axes in lower-left position
            axins = inset_axes(ax, width="42%", height="35%", loc='center',
                              bbox_to_anchor=(0.25, 0.10, 0.9, 0.9),
                              bbox_transform=ax.transAxes)
            
            # Re-plot the same data in the inset, focusing on step 400+
            for lr in sorted(self.config.learning_rates, reverse=True):
                cmap = lr_colormaps.get(lr, plt.cm.Greys)
                
                lr_runs = self.data.filter_runs(
                    metadata,
                    model_size=model_size,
                    method='lora',
                    learning_rate=lr
                )
                
                if len(lr_runs) == 0:
                    continue
                
                available_ranks = sorted(lr_runs['lora_rank'].dropna().unique())
                plot_ranks = [r for r in rep_ranks if r in available_ranks]
                
                if len(plot_ranks) == 0:
                    continue
                
                n_ranks = len(plot_ranks)
                
                for i, rank in enumerate(plot_ranks):
                    runs = lr_runs[lr_runs['lora_rank'] == rank]
                    
                    if len(runs) == 0:
                        continue
                    
                    run_name = runs.iloc[0]['name']
                    history = self.data.load_run_history(run_name)
                    
                    if history is None or 'train/loss' not in history.columns:
                        continue
                    
                    steps = history['_step'].values
                    losses = history['train/loss'].values
                    
                    # Filter for last 100 steps
                    if len(steps) <= 100:
                        steps_zoom = steps
                        losses_zoom = losses
                    else:
                        steps_zoom = steps[-100:]
                        losses_zoom = losses[-100:]
                    
                    if len(steps_zoom) == 0:
                        continue
                    
                    color_intensity = 0.25 + 0.60 * (i / max(n_ranks - 1, 1))
                    color = cmap(color_intensity)
                    
                    linewidth = 1.0 if rank in [64, 256, 512] else 0.8
                    
                    axins.plot(steps_zoom, losses_zoom, color=color, linestyle='-',
                              linewidth=linewidth, alpha=0.85)
            
            # Plot Full FT in inset (10x lower LRs with matching colors)
            if show_full_ft:
                for lora_lr, (ft_lr, ft_color, ft_linestyle) in lora_to_fullft_mapping.items():
                    full_runs = self.data.filter_runs(
                        metadata,
                        model_size=model_size,
                        method='full',
                        learning_rate=ft_lr  # Use 10x lower LR
                    )
                    
                    if len(full_runs) > 0:
                        run_name = full_runs.iloc[0]['name']
                        history = self.data.load_run_history(run_name)
                        
                        if history is not None and 'train/loss' in history.columns:
                            steps = history['_step'].values
                            losses = history['train/loss'].values
                            
                            # Filter for last 100 steps
                            if len(steps) <= 100:
                                steps_zoom = steps
                                losses_zoom = losses
                            else:
                                steps_zoom = steps[-100:]
                                losses_zoom = losses[-100:]
                            
                            if len(steps_zoom) > 0:
                                axins.plot(steps_zoom, losses_zoom, color=ft_color,
                                          linewidth=1.4, linestyle=ft_linestyle, alpha=0.9, zorder=10)
            
            # Format inset
            axins.set_xlabel('Steps', fontsize=9)
            axins.set_ylabel('Loss', fontsize=9)
            axins.set_title('Final 100 Steps', fontsize=9, pad=5)
            axins.tick_params(labelsize=8)
            axins.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
            
            # Mark the zoomed region on main plot (inverted: loc1=3, loc2=1)
            mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="0.5", linestyle='--', linewidth=1)
        
        plt.tight_layout()
        
        filename = f"all_learning_curves_{model_size}.png"
        return self._save_figure(filename)
    
    def plot_model_comparison_unified(self) -> Path:
        """
        Create unified 2x2 panel comparing 8B vs 70B models.
        
        Compares at:
        - LR=1e-3, r=64 and r=256
        - LR=1e-4, r=64 and r=256
        
        Returns:
            Path to saved plot
        """
        metadata = self.data.load_all_runs_metadata()
        
        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Define configurations: (lr, rank, row, col)
        configs = [
            (1e-3, 64, 0, 0),
            (1e-3, 256, 0, 1),
            (1e-4, 64, 1, 0),
            (1e-4, 256, 1, 1),
        ]
        
        # Model colors
        model_colors = {
            '8B': '#2E86AB',   # Blue
            '70B': '#A23B72',  # Purple
        }
        
        for lr, rank, row, col in configs:
            ax = axes[row, col]
            
            for model_size in self.config.model_sizes:
                runs = self.data.filter_runs(
                    metadata,
                    model_size=model_size,
                    method='lora',
                    learning_rate=lr,
                    lora_rank=rank
                )
                
                if len(runs) == 0:
                    continue
                
                run_name = runs.iloc[0]['name']
                history = self.data.load_run_history(run_name)
                
                if history is None or 'train/loss' not in history.columns:
                    continue
                
                steps = history['_step'].values
                losses = history['train/loss'].values
                
                ax.plot(
                    steps, losses,
                    label=model_size,
                    linewidth=2.5,
                    alpha=0.8,
                    color=model_colors.get(model_size, None)
                )
            
            # Format subplot
            lr_str = f"{lr:.0e}".replace('e-0', 'e-')
            ax.set_title(f'LR={lr_str}, Rank={rank}', fontweight='bold', fontsize=14, pad=10)
            ax.set_xlabel('Training Steps', fontweight='bold', fontsize=12)
            ax.set_ylabel('Training Loss', fontweight='bold', fontsize=12)
            ax.legend(loc='best', framealpha=0.95, fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
        
        # Overall title
        fig.suptitle('8B vs 70B Model Comparison', fontweight='bold', fontsize=16, y=0.995)
        
        plt.tight_layout()
        
        filename = 'model_comparison_unified.png'
        return self._save_figure(filename)
    
    def plot_model_comparison(
        self,
        learning_rate: float,
        lora_rank: int
    ) -> Path:
        """
        Compare 8B vs 70B models at same hyperparameters.
        
        DEPRECATED: Use plot_model_comparison_unified() instead.
        
        Args:
            learning_rate: Learning rate to compare
            lora_rank: LoRA rank to compare
        
        Returns:
            Path to saved plot
        """
        metadata = self.data.load_all_runs_metadata()
        
        fig, ax = plt.subplots(figsize=self.config.default_figsize)
        
        for model_size in self.config.model_sizes:
            runs = self.data.filter_runs(
                metadata,
                model_size=model_size,
                method='lora',
                learning_rate=learning_rate,
                lora_rank=lora_rank
            )
            
            if len(runs) == 0:
                continue
            
            run_name = runs.iloc[0]['name']
            history = self.data.load_run_history(run_name)
            
            if history is None or 'train/loss' not in history.columns:
                continue
            
            steps = history['_step'].values
            losses = history['train/loss'].values
            
            ax.plot(steps, losses, label=model_size, linewidth=2.5, alpha=0.8)
        
        ax.set_xlabel('Training Steps', fontweight='bold')
        ax.set_ylabel('Training Loss', fontweight='bold')
        ax.set_title(
            f'Model Comparison (LR={learning_rate:.0e}, r={lora_rank})',
            fontweight='bold',
            pad=20
        )
        ax.legend(loc='best', framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        lr_str = f"{learning_rate:.0e}".replace('e-0', 'e-')
        filename = f"model_comparison_lr{lr_str}_r{lora_rank}.png"
        return self._save_figure(filename)

