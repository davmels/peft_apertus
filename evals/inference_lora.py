import argparse
import torch
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from processing import PROCESSING_STRATEGIES
from utils import load_dataset_my_way


def load_model_and_tokenizer(base_model_name_or_path, lora_adapter_path):
    """
    Loads the base model, applies LoRA adapter, and prepares the tokenizer.
    """
    print(f"Loading base model: {base_model_name_or_path}...")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load Base Model (using bfloat16 to prevent NaN errors)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load and Merge LoRA Adapter
    if lora_adapter_path:
        print(f"Loading LoRA adapters from: {lora_adapter_path}...")
        model = PeftModel.from_pretrained(model, lora_adapter_path)

        # Merging makes inference slightly faster and simpler
        model = model.merge_and_unload()

    model.eval()

    return model, tokenizer


def generate_text(
    model, tokenizer, prompt, max_length, max_new_tokens=200, temperature=0.7
):
    """
    Runs the generation for a single prompt.
    """
    inputs = tokenizer(
        prompt, return_tensors="pt", max_length=max_length, truncation=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    gen_ids = outputs[0, input_len:].cpu().tolist()  # for batch size 1

    # Decode and strip special tokens
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def main(args):
    selected_processor = PROCESSING_STRATEGIES[args.processing_function]
    print(f"Using processing strategy: {args.processing_function}")

    model, tokenizer = load_model_and_tokenizer(
        args.base_model_name_or_path, args.lora_adapter_path
    )

    print(f"Loading dataset: {args.dataset_name} ({args.split})")
    dataset = load_dataset_my_way(
        args.dataset_name, name=args.dataset_config, split=args.split
    )

    if args.end_index is None:
        args.end_index = len(dataset)

    dataset = dataset.select(range(args.start_index, args.end_index))

    print(f"Starting inference on {len(dataset)} rows...")
    generated_results = []

    for idx, row in enumerate(tqdm(dataset, desc="Generating")):
        prompt = selected_processor(row, args.input_col, tokenizer)
        if idx == 0:
            print("Sample prompt:")
            print(prompt)

        try:
            output = generate_text(
                model,
                tokenizer,
                prompt,
                args.max_length,
                args.max_new_tokens,
                args.temperature,
            )
        except Exception as e:
            print(f"Error generating for row: {e}")
            output = "ERROR"

        generated_results.append(output)

    print("Inference complete. Saving results...")

    dataset = dataset.add_column(args.output_col, generated_results)

    dataset.save_to_disk(args.output_dir)
    print(f"Saved processed dataset to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LoRA Inference on a Dataset")

    # Model Arguments
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default="swiss-ai/Apertus-70B-Instruct-2509",
        help="HuggingFace Model ID",
    )
    parser.add_argument(
        "--lora_adapter_path", type=str, help="Path to the saved LoRA adapter"
    )

    # Dataset Arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name (e.g., 'wikitext')",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Dataset config name (optional)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (train/test/validation)",
    )
    parser.add_argument(
        "--input_col",
        type=str,
        default="text",
        help="Column name containing input text",
    )
    parser.add_argument(
        "--output_col",
        type=str,
        default="generated_response",
        help="Name of the new column to add",
    )

    # New Processing Argument
    parser.add_argument(
        "--processing_function",
        type=str,
        default="default",
        choices=list(PROCESSING_STRATEGIES.keys()),
        help=f"Choose how to process input rows. Options: {list(PROCESSING_STRATEGIES.keys())}",
    )
    parser.add_argument(
        "--max_length", type=int, default=4096, help="Maximum input length"
    )

    # Generation Arguments
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./inference_results",
        help="Directory to save the final dataset",
    )

    parser.add_argument(
        "--start_index", type=int, default=0, help="Starting index for processing"
    )
    parser.add_argument(
        "--end_index", type=int, default=None, help="Ending index for processing"
    )

    args = parser.parse_args()

    main(args)
