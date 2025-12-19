#!/usr/bin/env python3
"""
Fetch experiment metadata from WandB.

This script connects to the WandB API and fetches run metadata,
including configurations and summary metrics. It exports the data
to CSV files for subsequent analysis.

Usage:
    python 01_fetch_wandb_data.py
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis import Config, WandBClient, DataLoader
import pandas as pd
import json


def main():
    """Fetch and export WandB run metadata."""
    
    print("="*80)
    print("FETCHING WANDB DATA")
    print("="*80)
    
    # Initialize
    config = Config()
    client = WandBClient(config)
    data_loader = DataLoader(config)
    
    # Fetch all runs
    print(f"\n📡 Fetching runs from: {config.wandb_project_path}")
    runs = client.fetch_runs()
    print(f"✓ Found {len(runs)} total runs")
    
    # Export run summaries
    print(f"\n💾 Exporting run metadata...")
    summary_path = config.data_dir / 'runs_overview.csv'
    client.export_runs_summary(runs, summary_path)
    print(f"✓ Saved to: {summary_path}")
    
    # Parse all run names and create detailed metadata
    print(f"\n🔍 Parsing run names...")
    parsed_runs = []
    
    for run in runs:
        parsed = data_loader.parse_run_name(run.name)
        metadata = client.get_run_metadata(run)
        
        parsed_runs.append({
            'name': run.name,
            'id': run.id,
            'state': run.state,
            'created_at': run.created_at,
            'is_grid_run': parsed.get('is_grid_run', False),
            'learning_rate': parsed.get('learning_rate'),
            'lora_rank': parsed.get('lora_rank'),
            'method': parsed.get('method'),
            'model_size': parsed.get('model_size'),
            'final_loss': None  # Will be filled when downloading history
        })
    
    # Save parsed metadata
    parsed_df = pd.DataFrame(parsed_runs)
    parsed_path = config.data_dir / 'all_runs_parsed.csv'
    parsed_df.to_csv(parsed_path, index=False)
    print(f"✓ Parsed {len(parsed_runs)} runs")
    print(f"✓ Saved to: {parsed_path}")
    
    # Statistics
    print(f"\n📊 STATISTICS")
    print(f"{'='*80}")
    
    grid_runs = parsed_df[parsed_df['is_grid_run']]
    print(f"Total runs: {len(parsed_df)}")
    print(f"Grid search runs: {len(grid_runs)}")
    print(f"  - LoRA: {len(grid_runs[grid_runs['method'] == 'lora'])}")
    print(f"  - Full FT: {len(grid_runs[grid_runs['method'] == 'full'])}")
    print(f"Other runs: {len(parsed_df) - len(grid_runs)}")
    
    print(f"\nRun states:")
    for state, count in parsed_df['state'].value_counts().items():
        print(f"  - {state}: {count}")
    
    print(f"\nModel sizes (grid runs):")
    for model, count in grid_runs['model_size'].value_counts().items():
        print(f"  - {model}: {count}")
    
    # Save metadata summary
    metadata = {
        'total_runs': len(parsed_df),
        'grid_runs': len(grid_runs),
        'lora_runs': len(grid_runs[grid_runs['method'] == 'lora']),
        'full_ft_runs': len(grid_runs[grid_runs['method'] == 'full']),
        'states': parsed_df['state'].value_counts().to_dict(),
        'learning_rates': sorted(grid_runs['learning_rate'].dropna().unique().tolist()),
        'lora_ranks': sorted(grid_runs['lora_rank'].dropna().unique().astype(int).tolist()),
        'model_sizes': sorted(grid_runs['model_size'].dropna().unique().tolist())
    }
    
    metadata_path = config.data_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"\n✓ Saved metadata summary to: {metadata_path}")
    
    print(f"\n{'='*80}")
    print(f"✅ DATA FETCHING COMPLETE")
    print(f"{'='*80}")
    print(f"\nNext step: Run script 02_download_history.py")


if __name__ == '__main__':
    main()

