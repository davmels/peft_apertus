#!/usr/bin/env python3
"""
Evaluate a single LoRA checkpoint on LEXam MCQ using vLLM.
Usage:
    python eval_lora_checkpoint.py --lora-path /path/to/checkpoint --config config.yaml
"""
import os
import json
import argparse
import logging
import sys
import time
import datetime
import wandb
from typing import Any, Dict, List, Optional
from sklearn.metrics import classification_report
from collections import Counter
from dotenv import load_dotenv

# Local imports
from utils import load_yaml, setup_logger, format_prompt, parse_choice, get_choice_labels

# Load environment variables from .env file
load_dotenv()


def main():
    os.environ["WANDB_ENTITY"] = "lsaie-peft-apertus"
    os.environ["WANDB_PROJECT"] = "swiss_judgment_prediction"

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--lora-path", type=str, required=True, help="Path to LoRA adapter checkpoint")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--max-lora-rank", type=int, default=512, help="Max LoRA rank for vLLM")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    # Late imports
    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    rank = 0  # Single process for now
    logger = setup_logger(rank)

    # Parse checkpoint info from path
    lora_path = os.path.abspath(args.lora_path)
    checkpoint_name = os.path.basename(lora_path)
    parent_dir = os.path.basename(os.path.dirname(lora_path))
    
    # Extract run info (e.g., lr1e-3_r128_lora_8B/checkpoint-100)
    run_info = f"{parent_dir}/{checkpoint_name}" if "checkpoint" in checkpoint_name else checkpoint_name
    
    logger.info(f"Evaluating LoRA checkpoint: {run_info}")
    logger.info(f"Full path: {lora_path}")

    # Get base model from config
    base_model_id = cfg["model_id"]
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

    # WandB Init
    if not args.no_wandb:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{parent_dir}_{checkpoint_name}_{timestamp}"
        
        wandb.init(
            project=os.environ["WANDB_PROJECT"],
            entity=os.environ["WANDB_ENTITY"],
            name=run_name,
            config={
                **cfg,
                "lora_path": lora_path,
                "checkpoint_name": checkpoint_name,
                "run_info": run_info,
            }
        )

    logger.info("========== Config ==========")
    logger.info(json.dumps(cfg, indent=2))
    logger.info(f"Base model: {base_model_id}")
    logger.info(f"LoRA path: {lora_path}")

    # vLLM init with LoRA support
    start_init = time.time()
    llm = LLM(
        model=base_model_id,
        tensor_parallel_size=tp,
        max_model_len=max_model_len,
        seed=seed,
        trust_remote_code=True,
        gpu_memory_utilization=0.80,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
    )
    init_time = time.time() - start_init
    logger.info(f"vLLM Init Time: {init_time:.2f}s")
    
    if not args.no_wandb:
        wandb.log({"init_time_seconds": init_time})

    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )

    # Create LoRA request
    lora_request = LoRARequest(
        lora_name=run_info.replace("/", "_"),
        lora_int_id=1,
        lora_path=lora_path,
    )

    # Loop over subsets
    all_results = {}
    for subset in subsets:
        logger.info(f"Processing subset: {subset}")
        
        # Load dataset
        ds = load_dataset(dataset_repo, subset, split=split)

        # Build prompts + golds
        ds = ds.map(lambda x: {"prompt": format_prompt(x)}, desc=f"Formatting prompts for {subset}")
        prompts = list(ds["prompt"])
        
        golds = []
        for x in ds:
            labels = get_choice_labels(len(x["choices"]))
            golds.append(labels[int(x["gold"])])

        # Generate with LoRA adapter
        start_gen = time.time()
        outputs = llm.generate(prompts, sampling_params=sampling, lora_request=lora_request)
        gen_time = time.time() - start_gen
        
        preds = []
        predictions_data = []
        for i, out in enumerate(outputs):
            text = out.outputs[0].text
            pred = parse_choice(text, match_choice_regex)
            preds.append(pred)
            predictions_data.append([prompts[i], golds[i], pred, text, pred == golds[i]])

        # Metrics
        correct = sum([1 for p, g in zip(preds, golds) if p == g])
        total = len(golds)
        accuracy = correct / total if total > 0 else 0.0

        logger.info(f"Subset: {subset} | Accuracy: {accuracy:.4f} ({correct}/{total}) | Time: {gen_time:.2f}s")
        
        all_results[subset] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "generation_time": gen_time,
        }

        if not args.no_wandb:
            # Detailed Metrics
            unique_labels = sorted(list(set(golds + preds)))
            report = classification_report(golds, preds, output_dict=True, zero_division=0)
            
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
                f"{subset}/accuracy": accuracy,
                f"{subset}/correct": correct,
                f"{subset}/total": total,
                f"{subset}/generation_time": gen_time,
                f"{subset}/pred_distribution": wandb.Table(
                    data=[[k, v] for k, v in pred_counts.items()], 
                    columns=["Label", "Count"]
                ),
                f"{subset}/predictions": wandb.Table(
                    columns=["Prompt", "Gold", "Prediction", "Generated Text", "Correct"], 
                    data=predictions_data
                ),
            }
            
            if cm_plot:
                log_dict[f"{subset}/confusion_matrix"] = cm_plot
                
            wandb.log(log_dict)
            
            # Log classification report
            report_data = []
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    row = [label, metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support']]
                    report_data.append(row)
            
            wandb.log({
                f"{subset}/classification_report": wandb.Table(
                    data=report_data, 
                    columns=["Class", "Precision", "Recall", "F1", "Support"]
                )
            })

    # Print summary
    logger.info("========== Summary ==========")
    for subset, res in all_results.items():
        logger.info(f"{subset}: Accuracy={res['accuracy']:.4f} ({res['correct']}/{res['total']})")

    if not args.no_wandb:
        wandb.finish()
    
    return all_results


if __name__ == "__main__":
    main()
