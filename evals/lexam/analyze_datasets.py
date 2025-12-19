#!/usr/bin/env python3
import argparse
import os
import json
import numpy as np
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Local imports
from utils import load_yaml, format_prompt

load_dotenv()

def main():
    os.environ["WANDB_ENTITY"] = "lsaie-peft-apertus"
    os.environ["WANDB_PROJECT"] = "swiss_judgement_prediction"

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    
    model_id = cfg["model_id"]
    dataset_repo = cfg["dataset_repo"]
    subsets = cfg.get("subsets", ["mcq_4_choices"])
    split = cfg.get("split", "test")
    
    # WandB Init
    wandb_cfg = cfg.get("wandb", {})
    
    # Set env vars if not already set, to ensure consistency with eval script
    if "WANDB_ENTITY" not in os.environ and wandb_cfg.get("entity"):
        os.environ["WANDB_ENTITY"] = wandb_cfg.get("entity")
    if "WANDB_PROJECT" not in os.environ:
        os.environ["WANDB_PROJECT"] = wandb_cfg.get("project", "swiss_judgement_prediction")

    wandb.init(
        project=os.environ.get("WANDB_PROJECT"),
        entity=os.environ.get("WANDB_ENTITY"),
        name=f"dataset-analysis-{model_id.split('/')[-1]}",
        config=cfg,
        job_type="dataset_analysis"
    )

    print(f"Loading tokenizer for {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Falling back to a generic tokenizer (gpt2) just for length estimation if specific one fails.")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Table to store summary stats for all subsets
    summary_table_data = []
    summary_columns = ["Subset", "Min", "Max", "Mean", "Median", "Std", "P95", "P99", "Count"]

    for subset in subsets:
        print(f"Processing subset: {subset}")
        ds = load_dataset(dataset_repo, subset, split=split)
        
        # Format prompts
        prompts = [format_prompt(sample) for sample in ds]
        
        # Calculate lengths
        print(f"Tokenizing {len(prompts)} prompts...")
        encodings = tokenizer(prompts, add_special_tokens=False)
        lengths = [len(ids) for ids in encodings["input_ids"]]
        
        # Statistics
        stats = {
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths)),
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "std": float(np.std(lengths)),
            "p95": float(np.percentile(lengths, 95)),
            "p99": float(np.percentile(lengths, 99)),
            "count": len(lengths)
        }
        
        print(f"Stats for {subset}: {stats}")
        
        summary_table_data.append([
            subset, stats["min"], stats["max"], f"{stats['mean']:.2f}", 
            stats["median"], f"{stats['std']:.2f}", stats["p95"], stats["p99"], stats["count"]
        ])
        
        # Log stats to WandB
        wandb.log({
            f"{subset}/prompt_length_min": stats["min"],
            f"{subset}/prompt_length_max": stats["max"],
            f"{subset}/prompt_length_mean": stats["mean"],
            f"{subset}/prompt_length_median": stats["median"],
            f"{subset}/prompt_length_std": stats["std"],
            f"{subset}/prompt_length_p95": stats["p95"],
            f"{subset}/prompt_length_p99": stats["p99"],
        })
        
        # Create Histogram (WandB native)
        wandb.log({
            f"{subset}/prompt_length_dist": wandb.Histogram(lengths)
        })
        
        # Create Matplotlib Plot
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f"Prompt Length Distribution - {subset}")
        plt.xlabel("Token Count")
        plt.ylabel("Frequency")
        plt.axvline(stats['mean'], color='red', linestyle='dashed', linewidth=1, label=f"Mean: {stats['mean']:.1f}")
        plt.axvline(stats['median'], color='green', linestyle='dashed', linewidth=1, label=f"Median: {stats['median']:.1f}")
        plt.legend()
        plt.grid(axis='y', alpha=0.5)
        
        # Log image
        wandb.log({f"{subset}/prompt_length_plot": wandb.Image(plt)})
        plt.close()

    # Log summary table
    wandb.log({"dataset_stats_summary": wandb.Table(data=summary_table_data, columns=summary_columns)})

    wandb.finish()

if __name__ == "__main__":
    main()
