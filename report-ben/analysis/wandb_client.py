"""
WandB API client wrapper for fetching experiment data.
"""
import os
import time
from typing import List, Dict, Optional
from pathlib import Path

import wandb
import pandas as pd
from tqdm import tqdm

from .config import Config


class WandBClient:
    """Client for interacting with WandB API."""
    
    def __init__(self, config: Config):
        """
        Initialize WandB client.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self._api = None
    
    @property
    def api(self) -> wandb.Api:
        """Lazy-load WandB API instance."""
        if self._api is None:
            try:
                self._api = wandb.Api(timeout=60)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize WandB API: {e}\n"
                    "Please ensure you are logged in (run: wandb login)"
                ) from e
        return self._api
    
    def fetch_runs(self, state: Optional[str] = None) -> List:
        """
        Fetch runs from WandB project.
        
        Args:
            state: Filter by run state (e.g., 'finished', 'failed')
        
        Returns:
            List of WandB run objects
        """
        try:
            runs = self.api.runs(self.config.wandb_project_path)
            
            if state:
                runs = [r for r in runs if r.state == state]
            
            return runs
        except Exception as e:
            raise RuntimeError(f"Failed to fetch runs: {e}") from e
    
    def get_run_metadata(self, run) -> Dict:
        """
        Extract metadata from a WandB run.
        
        Args:
            run: WandB run object
        
        Returns:
            Dictionary with run metadata
        """
        config = {k: v for k, v in run.config.items() if not k.startswith('_')}
        summary = run.summary._json_dict
        
        return {
            'id': run.id,
            'name': run.name,
            'state': run.state,
            'created_at': run.created_at,
            'config': config,
            'summary': summary
        }
    
    def download_run_history(
        self,
        run,
        output_path: Path,
        overwrite: bool = False
    ) -> bool:
        """
        Download training history for a run.
        
        Args:
            run: WandB run object
            output_path: Where to save the CSV file
            overwrite: Whether to overwrite existing files
        
        Returns:
            True if downloaded successfully, False otherwise
        """
        if output_path.exists() and not overwrite:
            return False
        
        try:
            history = run.history()
            
            if history.empty:
                return False
            
            history.to_csv(output_path, index=False)
            return True
            
        except Exception as e:
            print(f"Warning: Failed to download {run.name}: {e}")
            return False
    
    def download_all_histories(
        self,
        runs: List,
        output_dir: Path,
        delay: float = 0.1,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Download histories for multiple runs.
        
        Args:
            runs: List of WandB run objects
            output_dir: Directory to save CSV files
            delay: Delay between downloads (for rate limiting)
            show_progress: Whether to show progress bar
        
        Returns:
            Dictionary with download statistics
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {'downloaded': 0, 'skipped': 0, 'failed': 0}
        
        iterator = tqdm(runs, desc="Downloading") if show_progress else runs
        
        for run in iterator:
            output_path = output_dir / f"{run.name}.csv"
            
            if output_path.exists():
                stats['skipped'] += 1
                continue
            
            success = self.download_run_history(run, output_path)
            
            if success:
                stats['downloaded'] += 1
            else:
                stats['failed'] += 1
            
            time.sleep(delay)
        
        return stats
    
    def export_runs_summary(
        self,
        runs: List,
        output_path: Path
    ) -> pd.DataFrame:
        """
        Export summary of all runs to CSV.
        
        Args:
            runs: List of WandB run objects
            output_path: Where to save the CSV file
        
        Returns:
            DataFrame with run summaries
        """
        summaries = []
        
        for run in tqdm(runs, desc="Exporting summaries"):
            metadata = self.get_run_metadata(run)
            summaries.append({
                'name': metadata['name'],
                'id': metadata['id'],
                'state': metadata['state'],
                'created_at': metadata['created_at'],
                'config': str(metadata['config']),
                'summary': str(metadata['summary'])
            })
        
        df = pd.DataFrame(summaries)
        df.to_csv(output_path, index=False)
        
        return df

