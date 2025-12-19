#!/usr/bin/env python3
"""
Download complete training history for all finished runs.

This script downloads the time-series training data (loss, accuracy, etc.)
for all finished experiments and saves them as CSV files.

Usage:
    python 02_download_history.py [--overwrite]
"""
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis import Config, WandBClient, DataLoader
import json


def main():
    """Download training histories for all finished runs."""
    
    parser = argparse.ArgumentParser(description='Download training histories')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing history files'
    )
    args = parser.parse_args()
    
    print("="*80)
    print("DOWNLOADING TRAINING HISTORIES")
    print("="*80)
    
    # Initialize
    config = Config()
    client = WandBClient(config)
    data_loader = DataLoader(config)
    
    # Load parsed metadata
    print(f"\n📂 Loading run metadata...")
    try:
        metadata = data_loader.load_all_runs_metadata()
    except FileNotFoundError:
        print("❌ Error: Run metadata not found.")
        print("   Please run 01_fetch_wandb_data.py first.")
        sys.exit(1)
    
    # Filter finished runs
    finished = metadata[metadata['state'] == 'finished']
    grid_runs = finished[finished['is_grid_run']]
    other_runs = finished[~finished['is_grid_run']]
    
    print(f"✓ Found {len(finished)} finished runs:")
    print(f"  - Grid search: {len(grid_runs)}")
    print(f"  - Other: {len(other_runs)}")
    
    # Fetch actual run objects
    print(f"\n📡 Fetching run objects from WandB...")
    all_runs = client.fetch_runs(state='finished')
    
    # Separate into grid and other
    grid_run_names = set(grid_runs['name'].values)
    grid_run_objects = [r for r in all_runs if r.name in grid_run_names]
    other_run_objects = [r for r in all_runs if r.name not in grid_run_names]
    
    print(f"✓ Matched {len(grid_run_objects)} grid run objects")
    print(f"✓ Matched {len(other_run_objects)} other run objects")
    
    # Download grid search histories
    print(f"\n{'='*80}")
    print(f"DOWNLOADING GRID SEARCH HISTORIES")
    print(f"{'='*80}\n")
    
    grid_stats = client.download_all_histories(
        grid_run_objects,
        config.history_dir,
        delay=0.1,
        show_progress=True
    )
    
    print(f"\n✓ Grid runs:")
    print(f"  - Downloaded: {grid_stats['downloaded']}")
    print(f"  - Skipped (existing): {grid_stats['skipped']}")
    print(f"  - Failed: {grid_stats['failed']}")
    
    # Download other histories
    other_dir = config.data_dir / 'history_other'
    other_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"DOWNLOADING OTHER HISTORIES")
    print(f"{'='*80}\n")
    
    other_stats = client.download_all_histories(
        other_run_objects,
        other_dir,
        delay=0.1,
        show_progress=True
    )
    
    print(f"\n✓ Other runs:")
    print(f"  - Downloaded: {other_stats['downloaded']}")
    print(f"  - Skipped (existing): {other_stats['skipped']}")
    print(f"  - Failed: {other_stats['failed']}")
    
    # Compute final losses and update metadata
    print(f"\n📊 Computing final losses...")
    
    updated_losses = []
    for idx, row in grid_runs.iterrows():
        final_loss = data_loader.get_final_loss(row['name'])
        updated_losses.append({
            'name': row['name'],
            'final_loss': final_loss
        })
    
    # Update metadata file
    losses_df = pd.DataFrame(updated_losses)
    metadata = metadata.merge(
        losses_df,
        on='name',
        how='left',
        suffixes=('_old', '')
    )
    
    # Drop old final_loss column if it exists
    if 'final_loss_old' in metadata.columns:
        metadata = metadata.drop(columns=['final_loss_old'])
    
    metadata_path = config.data_dir / 'all_runs_parsed.csv'
    metadata.to_csv(metadata_path, index=False)
    print(f"✓ Updated metadata with final losses")
    
    # Create inventory
    print(f"\n💾 Creating download inventory...")
    
    import time
    
    inventory = {
        'download_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'grid_runs': {
            'total': len(grid_run_objects),
            'downloaded': grid_stats['downloaded'],
            'skipped': grid_stats['skipped'],
            'failed': grid_stats['failed']
        },
        'other_runs': {
            'total': len(other_run_objects),
            'downloaded': other_stats['downloaded'],
            'skipped': other_stats['skipped'],
            'failed': other_stats['failed']
        }
    }
    
    inventory_path = config.data_dir / 'download_inventory.json'
    with open(inventory_path, 'w') as f:
        json.dump(inventory, f, indent=2)
    print(f"✓ Saved inventory to: {inventory_path}")
    
    # Validate grid coverage
    print(f"\n{'='*80}")
    print(f"VALIDATING GRID COVERAGE")
    print(f"{'='*80}")
    
    validation = data_loader.validate_grid_coverage()
    
    print(f"\nExpected grid points: {validation['total_expected']}")
    print(f"Found: {validation['total_found']}")
    print(f"Coverage: {validation['coverage_percent']:.1f}%")
    
    if validation['missing']:
        print(f"\n⚠️  Missing {len(validation['missing'])} grid points:")
        for item in validation['missing'][:10]:
            print(f"  - {item['model']} {item['method']} LR={item['lr']:.0e} r={item['rank']}")
        if len(validation['missing']) > 10:
            print(f"  ... and {len(validation['missing']) - 10} more")
    else:
        print(f"\n✅ All expected grid points have data!")
    
    print(f"\n{'='*80}")
    print(f"✅ DOWNLOAD COMPLETE")
    print(f"{'='*80}")
    print(f"\nNext step: Run script 03_generate_plots.py")


if __name__ == '__main__':
    import pandas as pd
    main()

