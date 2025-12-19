import os
import glob
import wandb
from datasets import load_from_disk
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import sys

# Add current directory to path to import from evaluate_generations
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from evaluate_generations import extract_label_swiss

ENTITY = "lsaie-peft-apertus"
PROJECT = "swiss_judgment_prediction"
PARENT_DIR = "/iopsstor/scratch/cscs/dmelikidze/LSAIE/project/peft_apertus/evals/inference_results/inference2/"


def evaluate_dataset(dataset_path):
    try:
        dataset = load_from_disk(dataset_path)
    except Exception as e:
        print(f"Failed to load {dataset_path}: {e}")
        return None

    y_true = []
    y_pred = []

    # Check columns - adjust if your dataset uses different names
    label_col = "label"
    gen_col = "generated_answer"

    if label_col not in dataset.column_names or gen_col not in dataset.column_names:
        print(f"Columns missing in {dataset_path}. Available: {dataset.column_names}")
        # Try to guess if names are different? For now, just return None
        return None

    for row in dataset:
        true_label = row[label_col]
        gen_text = row[gen_col]
        pred_label = extract_label_swiss(gen_text)

        y_true.append(true_label)
        y_pred.append(pred_label)

    return accuracy_score(y_true, y_pred)


def main():
    print(f"Connecting to WandB project: {ENTITY}/{PROJECT}")
    try:
        api = wandb.Api()
    except Exception as e:
        print(f"Error connecting to WandB API: {e}")
        print("Please ensure you are logged in (wandb login) and have access.")
        return

    # Walk through the parent directory
    # We expect structure: PARENT_DIR / group_dir / run_dir

    group_dirs = [
        d for d in glob.glob(os.path.join(PARENT_DIR, "*")) if os.path.isdir(d)
    ]

    if not group_dirs:
        print(f"No group directories found in {PARENT_DIR}")
        return

    for group_dir in group_dirs:
        print(f"\nScanning group: {os.path.basename(group_dir)}")
        run_dirs = [
            d for d in glob.glob(os.path.join(group_dir, "*")) if os.path.isdir(d)
        ]

        if not run_dirs:
            print(f"  No run directories found in {group_dir}")
            continue

        for run_dir in tqdm(run_dirs, desc="Processing runs"):
            run_name = os.path.basename(run_dir)

            # 1. Calculate Accuracy
            accuracy = evaluate_dataset(run_dir)
            if accuracy is None:
                continue

            # 2. Find WandB Run
            # We search for runs with the display_name matching the folder name
            print(f"  Searching for run with name: '{run_name}' in {ENTITY}/{PROJECT}")
            runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"display_name": run_name})

            if len(runs) == 0:
                print(f"  WARNING: No WandB run found for name: '{run_name}'")
                continue

            # Update all matching runs (usually just one)
            for run in runs:
                print(f"  Found Run: {run.name} (ID: {run.id})")
                print(f"  Run URL: {run.url}")

                # Update Summary (for Overview tab and Table columns)
                run.summary["eval/accuracy"] = accuracy
                run.summary.update()
                print(f"  -> Updated summary 'eval/accuracy' to {accuracy:.4f}")

                # Also log to history so it appears in Charts
                # We resume the run, log the metric, then finish it
                resumed_run = wandb.init(
                    entity=ENTITY, project=PROJECT, id=run.id, resume="must"
                )
                # Log at the final step (use a high step number or get from history)
                resumed_run.log({"eval/accuracy": accuracy})
                resumed_run.finish()
                print(f"  -> Logged 'eval/accuracy' to history (Charts)")


if __name__ == "__main__":
    main()
