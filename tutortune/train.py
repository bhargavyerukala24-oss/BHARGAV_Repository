import json
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from peft import LoraConfig, get_peft_model
from config import BASE_MODEL, MAX_LENGTH
from validate_data import validate


def main():
    validate()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Use a CUDA GPU runtime for this training guide."
        )
    set_seed(42)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
    )
    # Resolve and save the actual base revision used for this run.
    revision = model.config._commit_hash
    if not revision:
        raise RuntimeError("Base revision could not be resolved.")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=revision)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()
    dataset = load_dataset(
        "json",
        data_files={
            "train": "data/train.jsonl",
            "validation": "data/validation.jsonl",
        },
    )

    def encode(row):
        full = tokenizer.apply_chat_template(
            row["messages"], tokenize=True, add_generation_prompt=False
        )
        prefix = tokenizer.apply_chat_template(
            row["messages"][:-1], tokenize=True, add_generation_prompt=True
        )
        # A different model/template may not share this exact prefix.
        if full[: len(prefix)] != prefix:
            raise ValueError("Chat prefix mismatch: inspect the template.")
        # Reject instead of silently truncating away a target answer.
        if len(full) > MAX_LENGTH or len(prefix) >= len(full):
            raise ValueError(
                "Example too long or assistant target is empty."
            )
        padding = MAX_LENGTH - len(full)
        return {
            "input_ids": full + [tokenizer.pad_token_id] * padding,
            "attention_mask": [1] * len(full) + [0] * padding,
            "labels": [-100] * len(prefix)
            + full[len(prefix) :]
            + [-100] * padding,
        }

    encoded = dataset.map(encode, remove_columns=["topic", "messages"])
    args = TrainingArguments(
        output_dir="runs",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=False,
        bf16=False,
        report_to="none",
        seed=42,
        label_names=["labels"],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        data_collator=default_data_collator,
    )
    trainer.train()
    model.save_pretrained("adapter")
    tokenizer.save_pretrained("adapter")
    Path("adapter/training_metadata.json").write_text(
        json.dumps(
            {
                "base_model": BASE_MODEL,
                "base_revision": revision,
                "max_length": MAX_LENGTH,
                "seed": 42,
                "method": "LoRA; q_proj/v_proj; assistant-only loss",
            },
            indent=2,
        )
    )
    print("Saved adapter and tokenizer. Evaluate before deployment.")


if __name__ == "__main__":
    main()
