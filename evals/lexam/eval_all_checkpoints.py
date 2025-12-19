#!/usr/bin/env python3
"""
Evaluate all checkpoints (checkpoint-467) across grid search results using vLLM.
Supports both LoRA and full fine-tuned models for 8B and 70B.

Usage:
    python eval_all_checkpoints.py --config config_lora.yaml --grid-dir /path/to/results/grid_8B_lora
    python eval_all_checkpoints.py --config config_lora.yaml --grid-dir /path/to/results/grid_8B_full --full-finetune
"""
import os
import json
import argparse
import logging
import sys
import time
import datetime
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
from sklearn.metrics import classification_report
from collections import Counter

# Conditional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Local imports
from utils import load_yaml, setup_logger, format_prompt, parse_choice, get_choice_labels


def discover_checkpoints(grid_dir: str, checkpoint_name: str = "checkpoint-467") -> List[Dict[str, str]]:
    """Discover all model folders and their checkpoints."""
    checkpoints = []
    grid_path = Path(grid_dir)
    
    if not grid_path.exists():
        raise ValueError(f"Grid directory does not exist: {grid_dir}")
    
    for model_folder in sorted(grid_path.iterdir()):
        if not model_folder.is_dir():
            continue
        
        checkpoint_path = model_folder / checkpoint_name
        if checkpoint_path.exists():
            # Parse model folder name for hyperparams (e.g., lr1e-3_r128_lora_8B)
            folder_name = model_folder.name
            checkpoints.append({
                "model_folder": folder_name,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_name": checkpoint_name,
            })
        else:
            print(f"Warning: {checkpoint_name} not found in {model_folder}")
    
    return checkpoints


def is_lora_checkpoint(checkpoint_path: str) -> bool:
    """Check if checkpoint is a LoRA adapter (has adapter_config.json)."""
    return os.path.exists(os.path.join(checkpoint_path, "adapter_config.json"))


def get_base_model_from_adapter(checkpoint_path: str) -> str:
    """Get base model path from LoRA adapter config."""
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    with open(adapter_config_path, "r") as f:
        config = json.load(f)
    return config.get("base_model_name_or_path", "")


def get_max_lora_rank(checkpoint_path: str) -> int:
    """Get LoRA rank from adapter config."""
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    with open(adapter_config_path, "r") as f:
        config = json.load(f)
    return config.get("r", 64)


def evaluate_checkpoint(
    llm,
    sampling,
    checkpoint_info: Dict[str, str],
    dataset,
    match_choice_regex: str,
    lora_request=None,
    logger=None,
) -> Dict[str, Any]:
    """Evaluate a single checkpoint on the dataset."""
    from utils import get_choice_labels
    
    prompts = list(dataset["prompt"])
    
    golds = []
    for x in dataset:
        labels = get_choice_labels(len(x["choices"]))
        golds.append(labels[int(x["gold"])])

    start_gen = time.time()
    if lora_request:
        outputs = llm.generate(prompts, sampling_params=sampling, lora_request=lora_request)
    else:
        outputs = llm.generate(prompts, sampling_params=sampling)
    gen_time = time.time() - start_gen
    
    preds = []
    predictions_data = []
    for i, out in enumerate(outputs):
        text = out.outputs[0].text
        pred = parse_choice(text, match_choice_regex)
        preds.append(pred)
        predictions_data.append([prompts[i], golds[i], pred, text, pred == golds[i]])

    correct = sum([1 for p, g in zip(preds, golds) if p == g])
    total = len(golds)
    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "generation_time": gen_time,
        "preds": preds,
        "golds": golds,
        "predictions_data": predictions_data,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--grid-dir", type=str, required=True, help="Path to grid results directory")
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint-467", help="Checkpoint folder name")
    parser.add_argument("--output-csv", type=str, default=None, help="Output CSV file for results")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--max-lora-rank", type=int, default=512, help="Max LoRA rank for vLLM")
    parser.add_argument("--full-finetune", action="store_true", help="Evaluate full fine-tuned models (not LoRA)")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path (for full fine-tune with broken tokenizer)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--single-model", type=str, default=None, help="Evaluate only this model folder")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    
    # Late imports
    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    
    logger = setup_logger(0)
    
    # Discover checkpoints
    checkpoints = discover_checkpoints(args.grid_dir, args.checkpoint_name)
    
    if args.single_model:
        checkpoints = [c for c in checkpoints if c["model_folder"] == args.single_model]
    
    if not checkpoints:
        logger.error("No checkpoints found!")
        return
    
    logger.info(f"Found {len(checkpoints)} checkpoints to evaluate")
    for cp in checkpoints:
        logger.info(f"  - {cp['model_folder']}/{cp['checkpoint_name']}")

    # Determine model type from first checkpoint
    first_checkpoint = checkpoints[0]["checkpoint_path"]
    is_lora = is_lora_checkpoint(first_checkpoint) and not args.full_finetune
    
    if is_lora:
        from vllm.lora.request import LoRARequest
        base_model = get_base_model_from_adapter(first_checkpoint)
        logger.info(f"LoRA mode - Base model: {base_model}")
    else:
        base_model = None
        logger.info("Full fine-tune mode")

    # Config params
    dataset_repo = cfg["dataset_repo"]
    subsets = cfg.get("subsets", ["mcq_4_choices"])
    split = cfg.get("split", "test")
    match_choice_regex = cfg.get("match_choice_regex", r"###([A-Z]+)###")
    max_model_len = int(cfg.get("max_model_len", 8192))
    max_new_tokens = int(cfg.get("max_new_tokens", 2048))
    tp = args.tensor_parallel_size if args.tensor_parallel_size is not None else int(cfg.get("tensor_parallel_size", 1))
    temperature = float(cfg.get("temperature", 0.0))
    top_p = float(cfg.get("top_p", 1.0))
    seed = int(cfg.get("seed", 0))

    # WandB init
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        os.environ["WANDB_ENTITY"] = "lsaie-peft-apertus"
        os.environ["WANDB_PROJECT"] = "swiss_judgment_prediction"
        
        grid_name = os.path.basename(args.grid_dir)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{grid_name}_{args.checkpoint_name}_{timestamp}"
        
        wandb.init(
            project=os.environ["WANDB_PROJECT"],
            entity=os.environ["WANDB_ENTITY"],
            name=run_name,
            config={
                **cfg,
                "grid_dir": args.grid_dir,
                "checkpoint_name": args.checkpoint_name,
                "num_checkpoints": len(checkpoints),
                "is_lora": is_lora,
                "base_model": base_model,
            }
        )

    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )

    # Load dataset once
    logger.info(f"Loading dataset: {dataset_repo}")
    datasets_by_subset = {}
    for subset in subsets:
        ds = load_dataset(dataset_repo, subset, split=split)
        ds = ds.map(lambda x: {"prompt": format_prompt(x)}, desc=f"Formatting prompts for {subset}")
        datasets_by_subset[subset] = ds
        logger.info(f"  {subset}: {len(ds)} samples")

    # Results storage
    all_results = []
    
    # For LoRA: Initialize vLLM once with base model
    if is_lora:
        logger.info(f"Initializing vLLM with base model: {base_model}")
        start_init = time.time()
        llm = LLM(
            model=base_model,
            tensor_parallel_size=tp,
            max_model_len=max_model_len,
            seed=seed,
            trust_remote_code=True,
            gpu_memory_utilization=0.80,
            enable_lora=True,
            max_lora_rank=args.max_lora_rank,
        )
        init_time = time.time() - start_init
        logger.info(f"vLLM init time: {init_time:.2f}s")
        
        # Evaluate each checkpoint with LoRA adapter
        for idx, cp in enumerate(checkpoints):
            model_folder = cp["model_folder"]
            checkpoint_path = cp["checkpoint_path"]
            
            logger.info(f"\n[{idx+1}/{len(checkpoints)}] Evaluating: {model_folder}")
            
            lora_request = LoRARequest(
                lora_name=model_folder.replace("/", "_"),
                lora_int_id=idx + 1,
                lora_path=checkpoint_path,
            )
            
            for subset in subsets:
                ds = datasets_by_subset[subset]
                result = evaluate_checkpoint(
                    llm, sampling, cp, ds, match_choice_regex,
                    lora_request=lora_request, logger=logger
                )
                
                logger.info(f"  {subset}: Accuracy={result['accuracy']:.4f} ({result['correct']}/{result['total']})")
                
                result_row = {
                    "model_folder": model_folder,
                    "checkpoint": args.checkpoint_name,
                    "subset": subset,
                    "accuracy": result["accuracy"],
                    "correct": result["correct"],
                    "total": result["total"],
                    "generation_time": result["generation_time"],
                }
                all_results.append(result_row)
                
                if use_wandb:
                    # Detailed metrics like eval_lexam_vllm
                    preds = result["preds"]
                    golds = result["golds"]
                    predictions_data = result["predictions_data"]
                    
                    unique_labels = sorted(list(set(golds + preds)))
                    report = classification_report(golds, preds, output_dict=True, zero_division=0)
                    
                    # Confusion matrix
                    try:
                        label_to_id = {l: i for i, l in enumerate(unique_labels)}
                        golds_ids = [label_to_id[l] for l in golds]
                        preds_ids = [label_to_id[l] for l in preds]
                        cm_plot = wandb.plot.confusion_matrix(
                            probs=None,
                            y_true=golds_ids,
                            preds=preds_ids,
                            class_names=unique_labels
                        )
                    except Exception as e:
                        logger.warning(f"Could not create WandB confusion matrix: {e}")
                        cm_plot = None
                    
                    pred_counts = Counter(preds)
                    
                    log_dict = {
                        f"{model_folder}/{subset}/accuracy": result["accuracy"],
                        f"{model_folder}/{subset}/correct": result["correct"],
                        f"{model_folder}/{subset}/total": result["total"],
                        f"{model_folder}/{subset}/generation_time": result["generation_time"],
                        f"{model_folder}/{subset}/pred_distribution": wandb.Table(
                            data=[[k, v] for k, v in pred_counts.items()],
                            columns=["Label", "Count"]
                        ),
                        f"{model_folder}/{subset}/predictions": wandb.Table(
                            columns=["Prompt", "Gold", "Prediction", "Generated Text", "Correct"],
                            data=predictions_data
                        ),
                    }
                    
                    if cm_plot:
                        log_dict[f"{model_folder}/{subset}/confusion_matrix"] = cm_plot
                    
                    wandb.log(log_dict)
                    
                    # Classification report table
                    report_data = []
                    for label, metrics in report.items():
                        if isinstance(metrics, dict):
                            row = [label, metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support']]
                            report_data.append(row)
                    
                    wandb.log({
                        f"{model_folder}/{subset}/classification_report": wandb.Table(
                            data=report_data,
                            columns=["Class", "Precision", "Recall", "F1", "Support"]
                        )
                    })
    
    else:
        # Full fine-tune: need to reload model for each checkpoint
        # Use tokenizer from config or args (checkpoint tokenizers may be broken)
        tokenizer_path = args.tokenizer if args.tokenizer else cfg.get("model_id")
        logger.info(f"Using tokenizer from: {tokenizer_path}")
        
        for idx, cp in enumerate(checkpoints):
            model_folder = cp["model_folder"]
            checkpoint_path = cp["checkpoint_path"]
            
            logger.info(f"\n[{idx+1}/{len(checkpoints)}] Evaluating: {model_folder}")
            logger.info(f"  Loading model from: {checkpoint_path}")
            
            start_init = time.time()
            llm = LLM(
                model=checkpoint_path,
                tokenizer=tokenizer_path,
                tensor_parallel_size=tp,
                max_model_len=max_model_len,
                seed=seed,
                trust_remote_code=True,
                gpu_memory_utilization=0.80,
            )
            init_time = time.time() - start_init
            logger.info(f"  Model load time: {init_time:.2f}s")
            
            for subset in subsets:
                ds = datasets_by_subset[subset]
                result = evaluate_checkpoint(
                    llm, sampling, cp, ds, match_choice_regex,
                    lora_request=None, logger=logger
                )
                
                logger.info(f"  {subset}: Accuracy={result['accuracy']:.4f} ({result['correct']}/{result['total']})")
                
                result_row = {
                    "model_folder": model_folder,
                    "checkpoint": args.checkpoint_name,
                    "subset": subset,
                    "accuracy": result["accuracy"],
                    "correct": result["correct"],
                    "total": result["total"],
                    "generation_time": result["generation_time"],
                }
                all_results.append(result_row)
                
                if use_wandb:
                    # Detailed metrics like eval_lexam_vllm
                    preds = result["preds"]
                    golds = result["golds"]
                    predictions_data = result["predictions_data"]
                    
                    unique_labels = sorted(list(set(golds + preds)))
                    report = classification_report(golds, preds, output_dict=True, zero_division=0)
                    
                    # Confusion matrix
                    try:
                        label_to_id = {l: i for i, l in enumerate(unique_labels)}
                        golds_ids = [label_to_id[l] for l in golds]
                        preds_ids = [label_to_id[l] for l in preds]
                        cm_plot = wandb.plot.confusion_matrix(
                            probs=None,
                            y_true=golds_ids,
                            preds=preds_ids,
                            class_names=unique_labels
                        )
                    except Exception as e:
                        logger.warning(f"Could not create WandB confusion matrix: {e}")
                        cm_plot = None
                    
                    pred_counts = Counter(preds)
                    
                    log_dict = {
                        f"{model_folder}/{subset}/accuracy": result["accuracy"],
                        f"{model_folder}/{subset}/correct": result["correct"],
                        f"{model_folder}/{subset}/total": result["total"],
                        f"{model_folder}/{subset}/generation_time": result["generation_time"],
                        f"{model_folder}/{subset}/pred_distribution": wandb.Table(
                            data=[[k, v] for k, v in pred_counts.items()],
                            columns=["Label", "Count"]
                        ),
                        f"{model_folder}/{subset}/predictions": wandb.Table(
                            columns=["Prompt", "Gold", "Prediction", "Generated Text", "Correct"],
                            data=predictions_data
                        ),
                    }
                    
                    if cm_plot:
                        log_dict[f"{model_folder}/{subset}/confusion_matrix"] = cm_plot
                    
                    wandb.log(log_dict)
                    
                    # Classification report table
                    report_data = []
                    for label, metrics in report.items():
                        if isinstance(metrics, dict):
                            row = [label, metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support']]
                            report_data.append(row)
                    
                    wandb.log({
                        f"{model_folder}/{subset}/classification_report": wandb.Table(
                            data=report_data,
                            columns=["Class", "Precision", "Recall", "F1", "Support"]
                        )
                    })
            
            # Delete model to free GPU memory before loading next
            del llm
            import gc
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass

    # Save results to CSV
    if args.output_csv:
        output_path = args.output_csv
    else:
        grid_name = os.path.basename(args.grid_dir)
        output_path = os.path.join(args.grid_dir, f"eval_results_{args.checkpoint_name}.csv")
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model_folder", "checkpoint", "subset", "accuracy", "correct", "total", "generation_time"])
        writer.writeheader()
        writer.writerows(all_results)
    
    logger.info(f"\nResults saved to: {output_path}")

    # Print summary table
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    for subset in subsets:
        logger.info(f"\n{subset}:")
        subset_results = [r for r in all_results if r["subset"] == subset]
        subset_results.sort(key=lambda x: x["accuracy"], reverse=True)
        for r in subset_results:
            logger.info(f"  {r['model_folder']}: {r['accuracy']:.4f}")

    if use_wandb:
        # Log summary table
        wandb.log({
            "results_table": wandb.Table(
                columns=["model_folder", "checkpoint", "subset", "accuracy", "correct", "total"],
                data=[[r["model_folder"], r["checkpoint"], r["subset"], r["accuracy"], r["correct"], r["total"]] for r in all_results]
            )
        })
        wandb.finish()


if __name__ == "__main__":
    main()
