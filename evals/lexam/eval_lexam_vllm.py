#!/usr/bin/env python3
import os
import json
import argparse
import logging
import sys
import time
import datetime
import wandb
from typing import Any, Dict, List, Optional
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
from dotenv import load_dotenv

# Local imports
from utils import load_yaml, setup_logger, format_prompt, parse_choice

# Load environment variables from .env file
load_dotenv()

def infer_shard_args(num_shards: Optional[int], shard_id: Optional[int]) -> (int, int):
    if num_shards is not None and shard_id is not None:
        return num_shards, shard_id

    slurm_ntasks = os.environ.get("SLURM_NTASKS")
    slurm_procid = os.environ.get("SLURM_PROCID")
    if slurm_ntasks and slurm_procid:
        try:
            return int(slurm_ntasks), int(slurm_procid)
        except Exception:
            pass

    return 1, 0

def maybe_init_torch_distributed():
    try:
        import torch
        import torch.distributed as dist
    except Exception:
        return None

    if dist.is_available() and not dist.is_initialized():
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            dist.init_process_group(backend="nccl")
    return dist

def main():
    os.environ["WANDB_ENTITY"] = "lsaie-peft-apertus"
    os.environ["WANDB_PROJECT"] = "swiss_judgment_prediction"

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--pipeline-parallel-size", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging per sample")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    # Late imports
    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    # Optional distributed aggregation
    dist = maybe_init_torch_distributed()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    logger = setup_logger(rank)

    # Sharding
    num_shards, shard_id = infer_shard_args(args.num_shards, args.shard_id)
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"Invalid shard_id={shard_id} for num_shards={num_shards}")

    model_id = cfg["model_id"]
    dataset_repo = cfg["dataset_repo"]
    subsets = cfg.get("subsets", ["mcq_4_choices"])
    split = cfg.get("split", "test")
    match_choice_regex = cfg.get("match_choice_regex", r"###([A-Z]+)###")

    max_model_len = int(cfg.get("max_model_len", 8192))
    max_new_tokens = int(cfg.get("max_new_tokens", 2048))

    tp = args.tensor_parallel_size if args.tensor_parallel_size is not None else int(cfg.get("tensor_parallel_size", 1))
    pp = args.pipeline_parallel_size if args.pipeline_parallel_size is not None else int(cfg.get("pipeline_parallel_size", 1))

    # Update config with effective values for logging
    cfg["tensor_parallel_size"] = tp
    cfg["pipeline_parallel_size"] = pp

    temperature = float(cfg.get("temperature", 0.0))
    top_p = float(cfg.get("top_p", 1.0))
    seed = int(cfg.get("seed", 0))

    # WandB Init (only on rank 0)
    if rank == 0:
        wandb_cfg = cfg.get("wandb", {})
        
        base_name = wandb_cfg.get("name")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if base_name:
            run_name = f"{base_name}_{timestamp}"
        else:
            short_model_id = model_id.split('/')[-1]
            run_name = f"{short_model_id}_{split}_{timestamp}"

        wandb.init(
            project=os.environ["WANDB_PROJECT"],
            entity=os.environ["WANDB_ENTITY"],
            name=run_name,
            config=cfg
        )
        
        # Log devices
        try:
            import torch
            device_count = torch.cuda.device_count()
            devices = [torch.cuda.get_device_name(i) for i in range(device_count)]
            logger.info(f"Visible Devices: {devices}")
            wandb.config.update({"devices": devices})
        except Exception as e:
            logger.warning(f"Could not log devices: {e}")

    if rank == 0:
        logger.info("========== Config ==========")
        logger.info(json.dumps(cfg, indent=2))

    # vLLM init
    start_init = time.time()
    llm = LLM(
        model=model_id,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        max_model_len=max_model_len,
        seed=seed,
        trust_remote_code=True,
        gpu_memory_utilization=0.80,
    )
    init_time = time.time() - start_init
    if rank == 0:
        logger.info(f"vLLM Init Time: {init_time:.2f}s")
        wandb.log({"init_time_seconds": init_time})

    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )

    # Loop over subsets
    for subset in subsets:
        if rank == 0:
            logger.info(f"Processing subset: {subset}")
        
        # Load & shard dataset
        ds = load_dataset(dataset_repo, subset, split=split)
        if num_shards > 1:
            ds = ds.shard(num_shards=num_shards, index=shard_id, contiguous=True)

        # Build prompts + golds
        # Note: format_prompt now handles dynamic labels inside utils.py
        ds = ds.map(lambda x: {"prompt": format_prompt(x)}, desc=f"Formatting prompts for {subset}")
        prompts = list(ds["prompt"])
        
        # We need to reconstruct golds based on the dynamic labels
        # The dataset 'gold' is an index (0, 1, 2...). We need to map it to A, B, C...
        # We can use the helper from utils
        from utils import get_choice_labels
        
        golds = []
        for x in ds:
            labels = get_choice_labels(len(x["choices"]))
            golds.append(labels[int(x["gold"])])

        start_gen = time.time()
        outputs = llm.generate(prompts, sampling_params=sampling)
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

        if rank == 0:
            logger.info(f"Subset: {subset} | Accuracy: {accuracy:.2f} ({correct}/{total}) | Time: {gen_time:.2f}s")
            
            # Detailed Metrics
            # Ensure 'None' is in unique_labels if it appears in preds
            unique_labels = sorted(list(set(golds + preds)))
            
            report = classification_report(golds, preds, output_dict=True, zero_division=0)
            
            # WandB confusion matrix helper can be brittle with missing classes or 'None'
            # We will manually construct the confusion matrix plot data to be safe
            try:
                # Convert labels to indices explicitly to avoid WandB mapping errors
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
            
            # WandB Logging
            log_dict = {
                f"{subset}/accuracy": accuracy,
                f"{subset}/correct": correct,
                f"{subset}/total": total,
                f"{subset}/generation_time": gen_time,
                f"{subset}/pred_distribution": wandb.Table(data=[[k, v] for k, v in pred_counts.items()], columns=["Label", "Count"]),
                f"{subset}/predictions": wandb.Table(columns=["Prompt", "Gold", "Prediction", "Generated Text", "Correct"], data=predictions_data),
            }
            
            if cm_plot:
                log_dict[f"{subset}/confusion_matrix"] = cm_plot
                
            wandb.log(log_dict)
            
            # Log classification report as a table
            report_data = []
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    row = [label, metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support']]
                    report_data.append(row)
            
            wandb.log({
                f"{subset}/classification_report": wandb.Table(data=report_data, columns=["Class", "Precision", "Recall", "F1", "Support"])
            })

    if rank == 0:
        wandb.finish()

if __name__ == "__main__":
    main()
