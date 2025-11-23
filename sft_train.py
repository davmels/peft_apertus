# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
accelerate launch \
    --config_file configs/zero3.yaml \
    sft_train.py \
    --config configs/sft_lora.yaml \
    --model_name_or_path swiss-ai/Apertus-8B-Instruct-2509 \
    --processing_strategy swiss_judgement_prediction
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import transformers.modeling_utils
from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
)
from utils import load_and_process_dataset, CustomScriptArguments


def no_op(*args, **kwargs):
    pass


transformers.modeling_utils.PreTrainedModel._initialize_missing_keys = no_op


def main(script_args, training_args, model_args):
    # ====================================================
    #  WANDB SETUP
    # ====================================================
    # Configure logging to the specific Organization and Team
    os.environ["WANDB_ENTITY"] = "lsaie-peft-apertus"
    os.environ["WANDB_PROJECT"] = "swiss_judgement_prediction"

    # ------------------------
    # Load model & tokenizer
    # ------------------------
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # --------------
    # Load & Process Dataset
    # --------------
    dataset = load_and_process_dataset(script_args)

    # -------------
    # Train model
    # -------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split]
        if training_args.eval_strategy != "no"
        else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()
    print("Saving model to:", training_args.output_dir)
    trainer.save_model(training_args.output_dir)
    print("Model saved.")
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    # Use CustomScriptArguments instead of the default ScriptArguments
    parser = TrlParser((CustomScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args)
