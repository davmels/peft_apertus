from datasets import load_dataset, load_from_disk
from trl import ScriptArguments
from dataclasses import dataclass, field
import os
import wandb


@dataclass
class CustomScriptArguments(ScriptArguments):
    """
    Extends standard ScriptArguments to include data processing specific args.
    """

    processing_strategy: str = field(
        default="default",
        metadata={
            "help": "The name of the preprocessing strategy to apply (e.g., 'swiss_judgment_prediction')."
        },
    )
    from_disk: bool = field(
        default=False,
        metadata={
            "help": "Whether to load the dataset from disk (True) or from the hub (False)."
        },
    )
    debug_mode: bool = field(
        default=False,
        metadata={"help": "Whether to run in debug mode with limited data."},
    )


def format_swiss_judgment(example):
    """
    Converts raw rows (facts + label) into a single 'messages' column.
    """

    label_map = {0: "dismissal", 1: "approval"}

    raw_label = example.get("label", "")
    label_text = label_map.get(raw_label, str(raw_label))

    prompt_template = """You are a legal-analysis assistant. You will be given a Swiss Federal Supreme Court case along with its metadata and the full text of the opinion. Your task is to classify the outcome of the case.

TASK:
Return exactly one label: either "approval" or "dismissal". Output only one of these two words, with no additional text.

INSTRUCTIONS:
1. Carefully read the metadata below.
2. Carefully read the full case text.
3. Determine whether the correct outcome is "approval" or "dismissal".
4. Your final answer must be exactly one of these two labels:
   - approval
   - dismissal
5. Do not output anything else: no explanations, no justification, no punctuation, no commentary.

--- METADATA ---
Year: {year}
Language: {language}
Source language: {source_language}
Region: {region}
Canton: {canton}
Legal area: {legal_area}

Below is the complete case text.

<BEGIN CASE TEXT>
{text}
<END CASE TEXT>"""

    chat_content = [
        {
            "role": "user",
            "content": prompt_template.format(
                year=example.get("year", "N/A"),
                language=example.get("language", "N/A"),
                source_language=example.get("source_language", "N/A"),
                region=example.get("region", "N/A"),
                canton=example.get("canton", "N/A"),
                legal_area=example.get("legal_area", "N/A"),
                text=example.get("text", ""),
            ),
        },
        {"role": "assistant", "content": label_text},
    ]

    # --- KEY CHANGE 1: Return a dict with the column name 'messages' ---
    return {"messages": chat_content}


def format_default(example):
    """
    Default pass-through. Ensures a 'text' column exists.
    """
    if "text" not in example:
        return {"text": str(example)}
    return example


PROCESSING_STRATEGIES = {
    "swiss_judgment_prediction": format_swiss_judgment,
    "default": format_default,
}


def load_and_process_dataset(script_args):
    print(f"Loading dataset: {script_args.dataset_name}")
    if script_args.from_disk:
        dataset = load_from_disk(script_args.dataset_name)
    else:
        dataset = load_dataset(
            script_args.dataset_name,
            name=script_args.dataset_config,  # download_mode="force_redownload"
        )

    strategy_name = script_args.processing_strategy
    if strategy_name not in PROCESSING_STRATEGIES:
        raise ValueError(
            f"Unknown processing strategy: {strategy_name}. Available: {list(PROCESSING_STRATEGIES.keys())}"
        )

    print(f"Applying processing strategy: {strategy_name}")
    process_fn = PROCESSING_STRATEGIES[strategy_name]

    # Calculate columns to remove (feature columns, not splits)
    if hasattr(dataset, "column_names"):
        if isinstance(dataset.column_names, dict):
            # DatasetDict: grab cols from the first available split
            first_split = list(dataset.keys())[0]
            remove_cols = dataset[first_split].column_names
        else:
            # Single Dataset
            remove_cols = dataset.column_names
    else:
        remove_cols = None

    # Apply the mapping
    if strategy_name == "swiss_judgment_prediction":
        # 1. Map to new format (creates "messages" column, removes "year", "text", etc.)
        dataset = dataset.map(
            process_fn,
            num_proc=os.cpu_count(),
            remove_columns=remove_cols,
            # load_from_cache_file=False,
        )

        # 2. Remove unused SPLITS (e.g., 'test', 'validation')
        # Check if it is a dictionary (DatasetDict)
        if hasattr(dataset, "keys"):
            # Identify keys to delete
            splits_to_delete = [
                k for k in dataset.keys() if k != script_args.dataset_train_split
            ]

            if splits_to_delete:
                print(f"Removing unused splits: {splits_to_delete}")
                for split in splits_to_delete:
                    del dataset[split]

            # Safety check
            if script_args.dataset_train_split not in dataset:
                raise ValueError(
                    f"Target split '{script_args.dataset_train_split}' missing. "
                    f"Found: {list(dataset.keys())}"
                )

    else:
        # Default strategy
        dataset = dataset.map(
            process_fn,
            num_proc=os.cpu_count(),
            # load_from_cache_file=False
        )

    print("Dataset processing complete.")
    print(dataset)

    return dataset
