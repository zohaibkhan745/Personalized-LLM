"""
train_v4.py — Fine-tune Meta-Llama-3-8B-Instruct with QLoRA on a single RTX 4090.

This version implements Early Stopping to prevent overfitting, and opens up the MLP
layers (gate, up, down) with a rank of 16 to improve the model's actual reasoning capabilities.

Usage:
    python3 train_v4.py
"""

import json
import os
import sys
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import SFTConfig, SFTTrainer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR = "./llama3-math-tutor-adapter-v4"
TRAIN_FILE = "data/clean/train.jsonl"
VAL_FILE = "data/clean/validation.jsonl"
MAX_SEQ_LENGTH = 1024


def print_step(msg: str) -> None:
    """Print a clearly visible step marker."""
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# 1. Sanity checks
# ---------------------------------------------------------------------------
def sanity_checks() -> None:
    """Verify GPU availability and data files."""
    print_step("Step 1 / 9 — Sanity Checks")

    if not torch.cuda.is_available():
        sys.exit("ERROR: No CUDA GPU detected. A GPU is required for training.")

    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free_vram = (
        torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
    ) / (1024**3)
    print(f"GPU           : {gpu_name}")
    print(f"Total VRAM    : {total_vram:.1f} GB")
    print(f"Free VRAM     : {free_vram:.1f} GB")

    for path, label in [(TRAIN_FILE, "Training"), (VAL_FILE, "Validation")]:
        if not os.path.isfile(path):
            sys.exit(f"ERROR: {label} file not found at '{path}'")
    print("Data files    : ✓ both found")


# ---------------------------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------------------------
def load_data():
    """Load train and validation JSONL files."""
    print_step("Step 2 / 9 — Loading Data")

    dataset = load_dataset(
        "json",
        data_files={"train": TRAIN_FILE, "validation": VAL_FILE},
    )
    print(f"Train examples      : {len(dataset['train'])}")
    print(f"Validation examples : {len(dataset['validation'])}")
    return dataset


# ---------------------------------------------------------------------------
# 3. Tokenizer
# ---------------------------------------------------------------------------
def load_tokenizer():
    """Load and configure the tokenizer."""
    print_step("Step 3 / 9 — Loading Tokenizer")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # LLaMA-3 has no default pad token — reuse eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    print(f"Vocab size    : {tokenizer.vocab_size}")
    print(f"Pad token     : {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
    print(f"Padding side  : {tokenizer.padding_side}")
    return tokenizer


# ---------------------------------------------------------------------------
# 4 & 5. Quantization config + Model loading
# ---------------------------------------------------------------------------
def load_model():
    """Load the quantized model and prepare it for k-bit training."""
    print_step("Step 4 / 9 — Loading Model (4-bit QLoRA)")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    # Required for stable QLoRA — enables gradient computation on frozen params
    model = prepare_model_for_kbit_training(model)

    # Disable KV-cache (incompatible with gradient checkpointing)
    model.config.use_cache = False

    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        print(f"Model loaded on device(s): {device_map}")
    else:
        print(f"Model loaded on: {next(model.parameters()).device}")

    return model


# ---------------------------------------------------------------------------
# 6. LoRA  (v4 changes: added MLP layers, increased rank to 16)
# ---------------------------------------------------------------------------
def apply_lora(model):
    """Attach LoRA adapters to the model."""
    print_step("Step 5 / 9 — Applying LoRA Adapters")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameter summary
    trainable, total = model.get_nb_trainable_parameters()
    pct = 100 * trainable / total
    print(f"Total parameters      : {total:,}")
    print(f"Trainable parameters  : {trainable:,}")
    print(f"Trainable %           : {pct:.2f}%")
    return model


# ---------------------------------------------------------------------------
# 7. Training arguments  (SFTConfig = TrainingArguments + SFT-specific fields)
# ---------------------------------------------------------------------------
def get_training_args() -> SFTConfig:
    """Return the SFTConfig for this run."""
    return SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=15,             # High max epochs (Early Stopping will halt it safely)
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,   # effective batch size = 16
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        weight_decay=0.01,
        optim="paged_adamw_8bit",
        logging_steps=2,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,         # Automatically handled, but explicit is good
        bf16=True,
        seed=42,
        report_to="none",
        # SFT-specific fields
        max_length=MAX_SEQ_LENGTH,
        packing=False,
    )


# ---------------------------------------------------------------------------
# 8. Formatting function (apply_chat_template)
# ---------------------------------------------------------------------------
def make_formatting_fn(tokenizer):
    """Return a function that formats each example using the official
    LLaMA-3 Instruct chat template."""
    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
    return formatting_func


# ---------------------------------------------------------------------------
# 9. Build trainer & train
# ---------------------------------------------------------------------------
def train(model, tokenizer, dataset, training_args):
    """Create SFTTrainer, train, save, and evaluate."""
    print_step("Step 6 / 9 — Configuring SFTTrainer")

    formatting_fn = make_formatting_fn(tokenizer)

    # Implement Early Stopping to halt training if eval_loss stops improving for 3 evaluation checks.
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        formatting_func=formatting_fn,
        processing_class=tokenizer,
        callbacks=callbacks
    )

    # ---- Train ----
    print_step("Step 7 / 9 — Training Started")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print_step(f"Step 8 / 9 — Training Complete  ({elapsed/60:.1f} min)")

    # ---- Save adapter + tokenizer ----
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Adapter + tokenizer saved to: {OUTPUT_DIR}/")

    # ---- Final evaluation ----
    print_step("Step 9 / 9 — Final Evaluation")
    metrics = trainer.evaluate()
    print(f"Eval loss     : {metrics['eval_loss']:.4f}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Write results to JSON
    results_path = os.path.join(OUTPUT_DIR, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics written to: {results_path}")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    sanity_checks()
    dataset = load_data()
    tokenizer = load_tokenizer()
    model = load_model()
    model = apply_lora(model)
    training_args = get_training_args()
    train(model, tokenizer, dataset, training_args)
    print_step("All done! 🎉")


if __name__ == "__main__":
    main()
