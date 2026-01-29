from functools import partial
from pathlib import Path
import click
import os
import random
import numpy as np
import torch
from transformers import AutoTokenizer
from accelerate import PartialState

from accelerate.utils import wait_for_everyone

from scripts.slime.chat_template import CHAT_TEMPLATE
from trl import DPOTrainer, DPOConfig, CPOTrainer, CPOConfig, SFTTrainer, SFTConfig
from scripts.slime.setup import get_common_setup, get_dataset
from trl.trainer.slime_config import SlimeConfig
from trl.trainer.slime_trainer import SlimeTrainer

# Constants for deterministic splitting

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sft_formatting_func(batch, tokenizer):
    """
    Formats a batch of examples using the tokenizer's chat template.
    'batch' is a dictionary of lists (dataset columns).
    """
    output_texts = []
    for conversation in batch['chosen']:
        # 'conversation' is the list of dicts: [{'role': 'user', ...}, {'role': 'assistant', ...}]
        # tokenize=False returns the raw string with special tokens applied
        try:
            text = tokenizer.apply_chat_template([conversation], tokenize=False)
            output_texts.append(text)
        except Exception as e:
            # Fallback or error logging if a specific sample is malformed
            print(f"Error formatting sample: {e}")
            output_texts.append("")
    return output_texts


@click.command()
# Basic Configuration
@click.option("--method", type=click.Choice(["sft", "dpo", "slime", "cpo_simpo"]), default="dpo",
              help="Optimization method")
@click.option("--model-cache", type=click.Path(exists=True), default=Path("data/models"), help="Model cache path")
@click.option("--model-id", type=str, required=True, help="Path or ID of the model (Base for SFT, SFT-ckpt for others)")
@click.option("--dataset-path", type=str, required=True, help="Path to the binarized dataset")
@click.option("--output-dir", type=click.Path(exists=False), default=Path("./output"), help="Where to save the model")
# Shared Hyperparameters
@click.option("--lr", type=float, default=1e-4, help="Learning rate")
@click.option("--epochs", type=int, default=3, help="Num epochs")
@click.option("--batch-size", type=int, default=1, help="Per device train batch size")
@click.option("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
@click.option("--beta", type=float, default=0.1, help="Beta parameter (temperature)")
@click.option("--max-length", type=int, default=2048)
# SLiME Specific Hyperparams
@click.option("--slime-shift", type=float, default=1.25, help="SLiME rejected penalty shift")
@click.option("--slime-lambda-chosen", type=float, default=0.1)
@click.option("--slime-lambda-rejected", type=float, default=0.0005)
# SimPO Specific Hyperparams
@click.option("--simpo_gamma", type=float, default=2.0, help="Target margin for SimPO (cpo_simpo mode)")
def main(method, model_cache, model_id, epochs, dataset_path, output_dir, lr, batch_size, grad_accum,
         beta, max_length, slime_shift, slime_lambda_chosen,
         slime_lambda_rejected, simpo_gamma):
    # 1. Set Global Training Seed
    set_seed(0)
    model_cache = Path(model_cache)
    output_dir = Path(output_dir) / f"{Path(model_id).name}_{method}"

    # 2. Setup Model & Tokenizer
    model, tokenizer, peft_config = get_common_setup(model_cache, model_id)
    # instruct_tokenizer = AutoTokenizer.from_pretrained("data/models/Qwen3-4B-IT")
    # instruct_tokenizer = AutoTokenizer.from_pretrained("data/models/Llama-3.2-3B-IT")
    instruct_tokenizer = AutoTokenizer.from_pretrained("data/models/Gemma3-4B-IT")
    tokenizer.chat_template = instruct_tokenizer.chat_template
    # 3. Load & Split Dataset (CRITICAL STEP)
    # We load the full dataset first
    full_train_dataset, test_dataset = get_dataset(dataset_path)
    full_train_dataset = full_train_dataset.train_test_split(0.66, seed=0, shuffle=True)
    state = PartialState()
    if method == "sft":
        if state.is_main_process: print(f"Mode: SFT.")
        train_dataset = full_train_dataset["train"]
    else:
        if state.is_main_process: print(f"Mode: {method.upper()}.")
        train_dataset = full_train_dataset["test"]

    if state.is_main_process:
        print("Sample from active train_dataset:")
        print(train_dataset[0])

    # Common Config Args
    config_kwargs = {
        "num_train_epochs": epochs,
        "output_dir": str(output_dir),
        "learning_rate": lr,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "bf16": True,
        "logging_steps": 10,
        "logging_dir": str(output_dir / "logs"),
        "remove_unused_columns": False,
        "gradient_checkpointing": False,
        "eval_strategy": "steps",  # Use the new naming convention
        "eval_steps": 10000,
        "save_steps": 10000,
        "report_to": ["tensorboard"]
    }

    # --- TRAINER SELECTION ---

    if method == "sft":
        def tokenize_func(example):
            # 1. Full conversation (user + assistant)
            full_text = tokenizer.apply_chat_template(
                example["chosen"],
                tokenize=False
            )

            # 2. Prompt-only conversation (remove last assistant message)
            prompt_only = example["chosen"][:-1]

            prompt_text = tokenizer.apply_chat_template(
                prompt_only,
                tokenize=False,
                add_generation_prompt=True
            )

            # 3. Tokenize full sequence
            outputs = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding=False
            )

            # 4. Tokenize prompt-only sequence
            prompt_ids = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_length,
                padding=False
            )["input_ids"]

            prompt_len = len(prompt_ids)

            # 5. Mask prompt tokens
            labels = outputs["input_ids"].copy()
            labels[:prompt_len] = [-100] * prompt_len
            outputs["labels"] = labels

            return outputs

        column_names = train_dataset.column_names
        train_dataset = train_dataset.map(tokenize_func, remove_columns=column_names)
        test_dataset = test_dataset.map(tokenize_func, remove_columns=column_names)
        print(f"--- INITIALIZING SFT ---")
        # SFT config is slightly different (no beta, no dpo-specifics)
        args = SFTConfig(
            **config_kwargs,
            max_length=max_length,
            dataset_text_field="text",  # Used if dataset has a text column, otherwise formatting_func used
            packing=False  # Set to True if you want to pack sequences for speed
        )
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            processing_class=tokenizer,
            # formatting_func=partial(sft_formatting_func,tokenizer=tokenizer),
        )

    elif method == "dpo":
        print(f"--- INITIALIZING DPO ---")
        args = DPOConfig(
            **config_kwargs,
            loss_type="sigmoid",
            beta=beta,
            max_length=max_length,
            max_prompt_length=1024
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=None,  # TRL handles reference model creation automatically if None
            args=args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            processing_class=tokenizer
        )

    elif method == "slime":
        print(f"--- INITIALIZING SLIME ---")
        slime_config = SlimeConfig(
            **config_kwargs,
            beta=beta,
            max_length=max_length,
            max_prompt_length=1024,
            rejected_penalty_shift=slime_shift,
            center_lambda_chosen=slime_lambda_chosen,
            center_lambda_rejected=slime_lambda_rejected
        )
        trainer = SlimeTrainer(
            model=model,
            ref_model=None,
            args=slime_config,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            processing_class=tokenizer
        )

    elif method == "cpo_simpo":
        print(f"--- INITIALIZING CPO (SIMPO) ---")
        args = CPOConfig(
            **config_kwargs,
            beta=beta,
            max_length=max_length,
            max_prompt_length=1024,
            loss_type="simpo",
            simpo_gamma=simpo_gamma,
        )
        trainer = CPOTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            processing_class=tokenizer
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"Starting {method.upper()} Training...")

    trainer.train()
    wait_for_everyone()

    if state.is_main_process:
        # Save Model
        print("Saving model...")
        # Handle PEFT merging if applicable, or standard save
        if hasattr(trainer.model, "merge_and_unload"):
            try:
                merged_model = trainer.model.merge_and_unload(progressbar=True)
                model_to_save = merged_model
            except Exception as e:
                print(f"Could not merge and unload (might not be PEFT adapter): {e}")
                model_to_save = trainer.model
        else:
            model_to_save = trainer.model

        model_save_path = output_dir / "saved_checkpoint"
        model_to_save.save_pretrained(str(model_save_path))
        trainer.processing_class.save_pretrained(str(model_save_path))
        print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main()