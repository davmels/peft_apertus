import argparse
import sys
import os
import glob
from datasets import load_from_disk, load_dataset, concatenate_datasets

# 1. ADD accuracy_score HERE
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm


def extract_label_swiss(generated_text):
    """
    Parses 'dismissal' or 'approval' from the text.
    Returns integers: 0 for dismissal, 1 for approval.
    """
    if not isinstance(generated_text, str):
        return -1

    text = generated_text.lower()

    # Priority logic: Look for explicit keywords
    if "dismissal" in text or "dismissed" in text:
        return 0
    elif "approval" in text or "approved" in text:
        return 1

    return -1


def load_data_smart(dataset_path, from_disk=False, multi_folder=False, split="test"):
    """
    Handles loading from Hub, single disk folder, or multiple disk folders (concatenation).
    """
    # CASE A: Multi-folder pattern loading
    if multi_folder:
        print(f"Mode: Multi-folder scan. Prefix: {dataset_path}")

        # Ensure pattern ends with wildcard if user just gave a prefix
        search_pattern = dataset_path
        if not search_pattern.endswith("*"):
            search_pattern += "*"

        # Find folders
        folder_paths = sorted(glob.glob(search_pattern))
        # Filter to keep only actual directories
        folder_paths = [p for p in folder_paths if os.path.isdir(p)]

        if not folder_paths:
            raise ValueError(f"No folders found matching pattern: {search_pattern}")

        print(f"Found {len(folder_paths)} datasets to merge: {folder_paths}")

        # Load and Concatenate
        ds_list = [load_from_disk(p) for p in folder_paths]
        return concatenate_datasets(ds_list)

    # CASE B: Single folder from disk
    elif from_disk:
        print(f"Loading single dataset from disk: {dataset_path}")
        return load_from_disk(dataset_path)

    # CASE C: Hugging Face Hub
    else:
        print(f"Loading from Hub: {dataset_path}")
        return load_dataset(dataset_path, split=split)


def main(args):
    # 1. Load Dataset
    try:
        dataset = load_data_smart(
            dataset_path=args.dataset_path,
            from_disk=args.from_disk,
            multi_folder=args.multi_folder,
            split=args.split,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    print(f"Processing {len(dataset)} rows...")

    y_true = []
    y_pred = []
    parse_errors = 0

    # 2. Iterate and Extract
    for row in tqdm(dataset, desc="Evaluating"):
        # Get Ground Truth
        true_label = row.get(args.label_col)

        # Get Generation
        gen_text = row.get(args.gen_col, "")

        # Apply Logic
        pred_label = extract_label_swiss(gen_text)

        # Track stats
        if pred_label == -1:
            parse_errors += 1

        y_true.append(true_label)
        y_pred.append(pred_label)

    # 3. Report
    print("\n" + "=" * 30)
    print("RESULTS")
    print("=" * 30)
    print(f"Total Rows: {len(y_true)}")
    print(f"Parse Errors: {parse_errors} ({(parse_errors / len(y_true)) * 100:.2f}%)")

    # --- NEW: Calculate and Print Accuracy Explicitly ---
    # This treats -1 (parse errors) as incorrect predictions automatically
    acc = accuracy_score(y_true, y_pred)
    print(f"Global Accuracy: {acc:.4f}")
    print("-" * 30)

    print("\nClassification Report:")
    # 0 = Dismissal, 1 = Approval
    try:
        print(
            classification_report(
                y_true,
                y_pred,
                labels=[0, 1],
                target_names=["Dismissal (0)", "Approval (1)"],
                zero_division=0,
            )
        )
    except Exception as e:
        print(f"Standard report failed (likely due to type mismatch or -1s): {e}")
        print("Raw report including errors:")
        print(classification_report(y_true, y_pred, zero_division=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Swiss Judgement Generations")

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset (disk, hub name, or folder prefix)",
    )

    # Flags for loading mode
    parser.add_argument(
        "--from_disk",
        action="store_true",
        help="Load a single dataset from local disk",
    )
    parser.add_argument(
        "--multi_folder",
        action="store_true",
        help="Treat dataset_path as a prefix, load all matching folders, and concatenate",
    )

    parser.add_argument(
        "--split", type=str, default="test", help="Split if loading from Hub"
    )

    parser.add_argument(
        "--label_col",
        type=str,
        default="label",
        help="Column name for ground truth class labels",
    )
    parser.add_argument(
        "--gen_col",
        type=str,
        default="generated_answer",
        help="Column name for model generation",
    )

    args = parser.parse_args()

    main(args)
