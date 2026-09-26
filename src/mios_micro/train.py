"""SFT Training Pipeline for MiOS-Micro using TRL and PEFT LoRA.

Hardware-agnostic fine-tuner that adapts Qwen2.5-Coder-1.5B-Instruct to the
4-pillar MiOS dataset.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def detect_device() -> tuple[str, bool]:
    """Detect available hardware accelerator and 4-bit quantization support."""
    try:
        import torch
    except ImportError:
        return "cpu", False

    if torch.cuda.is_available():
        is_rocm = bool(getattr(torch.version, "hip", None))
        return ("rocm" if is_rocm else "cuda"), True
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", False
    return "cpu", False


def train(
    base_model: str,
    dataset_path: str,
    output_dir: str,
    epochs: int = 3,
    lr: float = 3e-4,
    batch_size: int = 4,
    grad_accum: int = 4,
    max_seq_len: int = 4096,
    lora_r: int = 16,
    lora_alpha: int = 32,
    dry_run: bool = False,
) -> dict:
    device, can_4bit = detect_device()
    
    plan = {
        "base_model": base_model,
        "dataset_path": dataset_path,
        "output_dir": output_dir,
        "device": device,
        "can_4bit": can_4bit,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "max_seq_len": max_seq_len,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
    }

    if not os.path.exists(dataset_path):
        return {"success": False, "error": f"Dataset file not found: {dataset_path}", "plan": plan}

    if dry_run:
        print("[mios-micro-train] Dry-run validation passed.")
        print(json.dumps(plan, indent=2))
        return {"success": True, "dry_run": True, "plan": plan}

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
    except ImportError as e:
        return {"success": False, "error": f"Missing required ML dependency: {e}", "plan": plan}

    print(f"[mios-micro-train] Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[mios-micro-train] Loading base model on device: {device}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        logging_steps=10,
        save_strategy="epoch",
        fp16=(device == "cuda" and not torch.cuda.is_bf16_supported()),
        bf16=(device == "cuda" and torch.cuda.is_bf16_supported()),
        optim="adamw_torch",
        report_to="none",
    )

    raw_dataset = load_dataset("json", data_files=dataset_path, split="train")

    trainer = SFTTrainer(
        model=model,
        train_dataset=raw_dataset,
        peft_config=lora_config,
        dataset_text_field="messages",
        max_seq_length=max_seq_len,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("[mios-micro-train] Starting training...")
    trainer.train()

    print(f"[mios-micro-train] Saving merged adapter to {output_dir}")
    trainer.save_model(output_dir)
    return {"success": True, "output_dir": output_dir, "plan": plan}


def main() -> int:
    parser = argparse.ArgumentParser(description="MiOS-Micro Training Harness")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct", help="Base HuggingFace model")
    parser.add_argument("--dataset", default="mios-micro-sft.jsonl", help="Path to SFT JSONL dataset")
    parser.add_argument("--output-dir", default="./output", help="Directory to save model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--dry-run", action="store_true", help="Validate plan without executing training")
    args = parser.parse_args()

    res = train(
        base_model=args.base_model,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        dry_run=args.dry_run,
    )

    print(json.dumps(res, indent=2))
    return 0 if res.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
