"""
Data loading and parsing utilities for LoRA experiments.
"""
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .config import Config


class DataLoader:
    """Loader for experiment data from CSV files and WandB exports."""
    
    def __init__(self, config: Config):
        """
        Initialize data loader.
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    @staticmethod
    def parse_run_name(name: str) -> Dict:
        """
        Extract hyperparameters from standardized run name.
        
        Expected format: lr{value}_r{rank}_{method}_{size}B
        Example: lr1e-4_r64_lora_8B
        
        Args:
            name: Run name string
        
        Returns:
            Dictionary with parsed parameters
        """
        pattern = r'lr([\d.e-]+)_r(\d+)_(lora|full)_(\d+)B'
        match = re.match(pattern, name)
        
        if not match:
            return {'is_grid_run': False, 'name': name}
        
        lr_str, rank, method, model_size = match.groups()
        
        return {
            'name': name,
            'learning_rate': float(lr_str.replace('e-', 'e-')),
            'lora_rank': int(rank) if method == 'lora' else None,
            'method': method,
            'model_size': f"{model_size}B",
            'is_grid_run': True
        }
    
    def load_run_history(self, run_name: str) -> Optional[pd.DataFrame]:
        """
        Load training history for a specific run.
        
        Args:
            run_name: Name of the run
        
        Returns:
            DataFrame with training history, or None if not found
        """
        filepath = self.config.history_dir / f"{run_name}.csv"
        
        if not filepath.exists():
            return None
        
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
            return None
    
    def load_all_runs_metadata(self) -> pd.DataFrame:
        """
        Load metadata for all runs from parsed CSV.
        
        Returns:
            DataFrame with all run metadata
        """
        filepath = self.config.data_dir / 'all_runs_parsed.csv'
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {filepath}\n"
                "Please run the data fetching script first."
            )
        
        return pd.read_csv(filepath)
    
    def filter_runs(
        self,
        df: pd.DataFrame,
        model_size: Optional[str] = None,
        method: Optional[str] = None,
        learning_rate: Optional[float] = None,
        lora_rank: Optional[int] = None,
        state: str = 'finished'
    ) -> pd.DataFrame:
        """
        Filter runs by various criteria.
        
        Args:
            df: DataFrame with run metadata
            model_size: Filter by model size (e.g., '8B')
            method: Filter by method ('lora' or 'full')
            learning_rate: Filter by learning rate
            lora_rank: Filter by LoRA rank
            state: Filter by run state
        
        Returns:
            Filtered DataFrame
        """
        mask = df['state'] == state
        
        if model_size:
            mask &= df['model_size'] == model_size
        
        if method:
            mask &= df['method'] == method
        
        if learning_rate is not None:
            mask &= df['learning_rate'] == learning_rate
        
        if lora_rank is not None:
            mask &= df['lora_rank'] == lora_rank
        
        return df[mask].copy()
    
    def get_final_loss(self, run_name: str) -> Optional[float]:
        """
        Extract final loss value from run history.
        
        Args:
            run_name: Name of the run
        
        Returns:
            Final loss value, or None if not available
        """
        history = self.load_run_history(run_name)
        
        if history is None or history.empty:
            return None
        
        # Try common column names for loss
        loss_columns = ['train/loss', 'loss', 'train_loss']
        
        for col in loss_columns:
            if col in history.columns:
                loss_series = history[col].dropna()
                if not loss_series.empty:
                    return float(loss_series.iloc[-1])
        
        return None
    
    def compute_learning_curve_stats(
        self,
        run_name: str
    ) -> Dict[str, float]:
        """
        Compute statistics from learning curve.
        
        Args:
            run_name: Name of the run
        
        Returns:
            Dictionary with statistics (final_loss, min_loss, etc.)
        """
        history = self.load_run_history(run_name)
        
        if history is None or history.empty:
            return {}
        
        # Find loss column
        loss_col = None
        for col in ['train/loss', 'loss', 'train_loss']:
            if col in history.columns:
                loss_col = col
                break
        
        if loss_col is None:
            return {}
        
        losses = history[loss_col].dropna()
        
        if losses.empty:
            return {}
        
        return {
            'final_loss': float(losses.iloc[-1]),
            'min_loss': float(losses.min()),
            'mean_loss': float(losses.mean()),
            'num_steps': len(losses)
        }
    
    def get_grid_search_results(self) -> pd.DataFrame:
        """
        Get complete grid search results with final losses.
        
        Returns:
            DataFrame with all grid search runs and their final losses
        """
        metadata = self.load_all_runs_metadata()
        grid_runs = metadata[metadata['is_grid_run']].copy()
        
        # Add final loss if not already present
        if 'final_loss' not in grid_runs.columns:
            grid_runs['final_loss'] = grid_runs['name'].apply(self.get_final_loss)
        
        return grid_runs
    
    def validate_grid_coverage(self) -> Dict[str, any]:
        """
        Validate that all expected grid points have data.
        
        Returns:
            Dictionary with validation results
        """
        grid_runs = self.get_grid_search_results()
        
        results = {
            'total_expected': 0,
            'total_found': 0,
            'missing': [],
            'coverage_by_model': {}
        }
        
        for model in self.config.model_sizes:
            for method in ['lora', 'full']:
                if method == 'lora':
                    ranks = self.config.lora_ranks
                else:
                    ranks = [1]  # Full FT doesn't have ranks
                
                for lr in self.config.learning_rates:
                    for rank in ranks:
                        results['total_expected'] += 1
                        
                        # Check if this run exists
                        mask = (
                            (grid_runs['model_size'] == model) &
                            (grid_runs['method'] == method) &
                            (grid_runs['learning_rate'] == lr)
                        )
                        
                        if method == 'lora':
                            mask &= grid_runs['lora_rank'] == rank
                        
                        if mask.sum() > 0:
                            results['total_found'] += 1
                        else:
                            results['missing'].append({
                                'model': model,
                                'method': method,
                                'lr': lr,
                                'rank': rank
                            })
        
        results['coverage_percent'] = (
            100 * results['total_found'] / results['total_expected']
            if results['total_expected'] > 0 else 0
        )
        
        return results

