import argparse
import torch
import os
from transformers import AutoTokenizer
from processing import PROCESSING_STRATEGIES
from utils import load_dataset_my_way

# vLLM imports
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def main(args):
    # 1. Select Processor
    selected_processor = PROCESSING_STRATEGIES[args.processing_function]
    print(f"Using processing strategy: {args.processing_function}")

    # 2. Load Dataset
    print(f"Loading dataset: {args.dataset_name} ({args.split})")
    dataset = load_dataset_my_way(
        args.dataset_name, name=args.dataset_config, split=args.split
    )

    if args.end_index is None:
        args.end_index = len(dataset)

    # Slice dataset based on start/end index
    dataset = dataset.select(range(args.start_index, args.end_index))
    print(f"Processing {len(dataset)} rows...")

    # 3. Load Tokenizer (Always uses base model path for tokenizer config)
    print(f"Loading tokenizer: {args.base_model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 4. Initialize vLLM Engine
    print("Initializing vLLM Engine...")
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs. Using tensor_parallel_size={num_gpus}")

    # --- LOGIC UPDATE START ---

    # Check if a LoRA/Model path is provided
    has_adapter_path = (
        args.lora_adapter_path is not None and len(args.lora_adapter_path) > 0
    )

    if args.load_adapter_as_model:
        # Case A: Treat the "adapter path" as a full merged model
        if not has_adapter_path:
            raise ValueError(
                "You set --load_adapter_as_model but did not provide --lora_adapter_path"
            )

        print(f"-> Loading FULL MODEL from: {args.lora_adapter_path}")
        model_path = args.lora_adapter_path
        enable_lora_flag = False
    else:
        # Case B: Standard Base Model + Optional Runtime LoRA
        print(f"-> Loading BASE MODEL from: {args.base_model_name_or_path}")
        model_path = args.base_model_name_or_path
        # Only enable LoRA if the path is provided AND we aren't loading it as a full model
        enable_lora_flag = has_adapter_path
        if enable_lora_flag:
            print(f"-> Runtime LoRA Enabled. Adapter: {args.lora_adapter_path}")
        else:
            print("-> LoRA Disabled.")

    # Construct engine arguments
    engine_args = {
        "model": model_path,
        "tokenizer": args.base_model_name_or_path,  # Tokenizer usually stays consistent
        "dtype": "bfloat16",
        "tensor_parallel_size": num_gpus,
        "max_model_len": args.max_length,
        "trust_remote_code": True,
        "gpu_memory_utilization": 0.85,
        "enable_lora": enable_lora_flag,
    }

    if enable_lora_flag:
        engine_args["max_lora_rank"] = 512

    # Initialize LLM
    llm = LLM(**engine_args)
    # --- LOGIC UPDATE END ---

    # 5. Define Sampling Parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )

    # Define LoRA request object if needed
    lora_req = None
    if enable_lora_flag:
        # Note: 'adapter' is just a name, '1' is the unique ID for this adapter
        lora_req = LoRARequest("adapter", 1, args.lora_adapter_path)

    # 6. Pre-process All Prompts
    print("Preparing prompts...")
    all_prompts = []

    for row in dataset:
        prompt = selected_processor(row, args.input_col, tokenizer)
        all_prompts.append(prompt)

    if len(all_prompts) > 0:
        print("\n--- Sample Prompt ---")
        print(all_prompts[0])
        print("---------------------\n")

    # 7. Generate All at Once
    print(
        f"Starting vLLM continuous batch generation for {len(all_prompts)} prompts..."
    )

    outputs = llm.generate(
        all_prompts,
        sampling_params,
        lora_request=lora_req,  # Pass None if enable_lora is False
        use_tqdm=True,
    )

    # 8. Extract Results
    generated_results = [output.outputs[0].text for output in outputs]

    # 9. Save Results
    print("Inference complete. Saving results...")
    dataset = dataset.add_column(args.output_col, generated_results)

    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    dataset.save_to_disk(args.output_dir)
    print(f"Saved processed dataset to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vLLM LoRA Inference")

    # Model Arguments
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default="swiss-ai/Apertus-70B-Instruct-2509",
        help="HuggingFace Model ID (Used as tokenizer source and default base model)",
    )

    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        default=None,
        help="Path to the LoRA adapter OR the full model if --load_adapter_as_model is set.",
    )

    # --- NEW FLAG ---
    parser.add_argument(
        "--load_adapter_as_model",
        action="store_true",
        help="If set, treats lora_adapter_path as a FULL model path (merged weights) instead of a LoRA adapter.",
    )
    # ----------------

    # Dataset Arguments
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--input_col", type=str, default="text")
    parser.add_argument("--output_col", type=str, default="generated_response")

    # Processing
    parser.add_argument(
        "--processing_function",
        type=str,
        default="default",
        choices=list(PROCESSING_STRATEGIES.keys()),
    )
    parser.add_argument("--max_length", type=int, default=4096 * 2 * 2)

    # Generation Arguments
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./inference_results",
    )
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)

    args = parser.parse_args()

    main(args)
