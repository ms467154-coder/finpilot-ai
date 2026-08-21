"""Fine-tuning driver for the Financial Advice Chatbot (Phase 5).

Loads the base model + tokenizer (``src/model/model_loader.py``), builds train /
val datasets from the Phase 3 formatted JSONL (``src/training/dataset.py``),
and fine-tunes with Hugging Face :class:`~transformers.Trainer` using a LoRA or
full fine-tune (``configs/training_config.yaml``).

Outputs:

* checkpoints / final adapter under ``training.output_dir`` (``models/fine_tuned/``);
* a per-step loss dump (``training.loss_curve_file``) plus a matplotlib loss
  curve (``training.loss_curve_image``) under ``logs/``.

Run::

    python src/training/train.py                       # full config-driven run
    python src/training/train.py --train-sample 96 --val-sample 32 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.model import model_loader  # noqa: E402
from src.training import dataset as train_dataset  # noqa: E402

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except Exception:  # pragma: no cover - peft is optional
    PEFT_AVAILABLE = False

from transformers import Trainer, TrainingArguments, set_seed  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = _ROOT / "configs" / "training_config.yaml"

DEFAULT_CONFIG: Dict[str, Any] = {
    "training": {
        "method": "lora",
        "system_prompt": None,
        "max_length": 512,
        "seed": 42,
        "num_train_epochs": 3,
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": False,
        "logging_steps": 5,
        "eval_strategy": "epoch",
        "eval_steps": 50,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "train_file": "data/processed/formatted/train.jsonl",
        "val_file": "data/processed/formatted/val.jsonl",
        "output_dir": "models/fine_tuned",
        "logging_dir": "logs",
        "loss_curve_file": "logs/training_losses.jsonl",
        "loss_curve_image": "logs/loss_curve.png",
    },
    "lora": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": "all-linear",
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
}


def load_config(path: Union[str, Path, None] = None) -> Dict[str, Any]:
    """Read + deep-merge :file:`configs/training_config.yaml` over the defaults."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    cfg: Dict[str, Any] = DEFAULT_CONFIG.copy()
    cfg["training"] = {**DEFAULT_CONFIG["training"]}
    if path.exists():
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cfg["training"].update(raw.get("training", {}))
        cfg["lora"] = {**cfg.get("lora", {}), **(raw.get("lora", {}))}
    return cfg


def _resolve(path: Union[str, Path]) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _ROOT / p


# ---------------------------------------------------------------------------
# Peft
# ---------------------------------------------------------------------------
def apply_lora(model, model_cfg: Mapping[str, Any]) -> torch.nn.Module:
    """Wrap ``model`` in a LoRA ``PeftModel`` using ``configs/training_config.yaml``."""
    if not PEFT_AVAILABLE:
        raise RuntimeError("peft is required for method='lora' (pip install peft)")
    lora_cfg = LoraConfig(
        r=model_cfg["r"],
        lora_alpha=model_cfg["lora_alpha"],
        lora_dropout=model_cfg["lora_dropout"],
        target_modules=model_cfg["target_modules"],
        bias=model_cfg["bias"],
        task_type=model_cfg["task_type"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Loss-curve logging
# ---------------------------------------------------------------------------
def dump_loss_curves(trainer: Trainer, cfg: Dict[str, Any]) -> None:
    """Write per-step loss log (JSONL) + matplotlib PNG to ``logs/``."""
    out_file = _resolve(cfg["training"]["loss_curve_file"])
    img_file = _resolve(cfg["training"]["loss_curve_image"])
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rows: list = []
    for lg in trainer.state.log_history:
        if "loss" not in lg and "eval_loss" not in lg:
            continue
        row = {"step": lg.get("step"), "epoch": lg.get("epoch")}
        if "loss" in lg:
            row["train_loss"] = lg["loss"]
        if "eval_loss" in lg:
            row["eval_loss"] = lg["eval_loss"]
        if "learning_rate" in lg:
            row["lr"] = lg["learning_rate"]
        rows.append(row)

    with open(out_file, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    logger.info("Wrote %d loss-curve rows -> %s", len(rows), out_file)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [r["step"] for r in rows]
        train_loss = [r["train_loss"] for r in rows if "train_loss" in r]
        train_steps = [r["step"] for r in rows if "train_loss" in r]
        eval_loss = [r["eval_loss"] for r in rows if "eval_loss" in r]
        eval_steps = [r["step"] for r in rows if "eval_loss" in r]

        if not train_loss and not eval_loss:
            logger.info("No loss values to plot; skipping image.")
            return
        plt.figure(figsize=(8, 4.5))
        if train_loss:
            plt.plot(train_steps, train_loss, marker="o", label="train_loss")
        if eval_loss:
            plt.plot(eval_steps, eval_loss, marker="s", label="eval_loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("Fine-tuning loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(img_file, dpi=110)
        plt.close()
        logger.info("Saved loss curve -> %s", img_file)
    except Exception as exc:  # noqa: BLE001 - plotting is best-effort
        logger.warning("Could not render loss curve image: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune the Financial Advice Chatbot base model.")
    p.add_argument("--config", default=None, help="Path to a training config YAML (default: configs/training_config.yaml)")
    p.add_argument("--method", choices=["lora", "full"], default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--system-prompt", default=None, help="Advisor persona string (overrides config)")
    p.add_argument("--train-sample", type=int, default=None, help="Use only first N train rows (validation runs)")
    p.add_argument("--val-sample", type=int, default=None, help="Use only first N val rows")
    p.add_argument("--quantization", default=None, help="4bit | 8bit | none for the base model load")
    p.add_argument("--dtype", default=None, help="Load-dtype for the base model (default: bfloat16)")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    tr = cfg["training"]

    if args.epochs is not None:
        tr["num_train_epochs"] = args.epochs
    if args.method is not None:
        tr["method"] = args.method
    if args.batch_size is not None:
        tr["batch_size"] = args.batch_size
    if args.lr is not None:
        tr["learning_rate"] = args.lr
    if args.max_length is not None:
        tr["max_length"] = args.max_length
    if args.system_prompt is not None:
        tr["system_prompt"] = args.system_prompt
    if args.seed is not None:
        tr["seed"] = args.seed

    set_seed(tr["seed"])
    logging.basicConfig(
        level=getattr(logging, str(tr.get("log_level", "info")).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    output_dir = _resolve(tr["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir = _resolve(tr["logging_dir"])
    logging_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Config: method=%s epochs=%d batch=%d lr=%g max_length=%d system_prompt=%r",
                tr["method"], tr["num_train_epochs"], tr["batch_size"], tr["learning_rate"],
                tr["max_length"], tr["system_prompt"])

    # model + tokenizer
    tokenizer = model_loader.load_tokenizer()
    model = model_loader.load_model(
        load_dtype=args.dtype or "bfloat16",
        quantization=args.quantization,
    )
    if args.quantization in ("4bit", "8bit"):
        model = prepare_model_for_kbit_training(model)
    if tr["method"] == "lora":
        model = apply_lora(model, cfg["lora"])
    logger.info("Model ready: %s", model_loader.get_model_info(model, base_model=model.name_or_path))

    # datasets
    train_ds = train_dataset.build_sft_dataset(
        _resolve(tr["train_file"]), tokenizer,
        max_length=tr["max_length"], system_prompt=tr["system_prompt"],
        sample=args.train_sample,
    )
    logger.info("Train samples: %d", len(train_ds))

    val_path = _resolve(tr["val_file"])
    val_ds = None
    if tr["eval_strategy"] != "no" and val_path.exists():
        val_ds = train_dataset.build_sft_dataset(
            val_path, tokenizer,
            max_length=tr["max_length"], system_prompt=tr["system_prompt"],
            sample=args.val_sample,
        )
        logger.info("Val samples:   %d", len(val_ds))

    collate_fn = train_dataset.make_collate_fn(pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)

    steps_per_epoch = int(math.ceil(len(train_ds) / (tr["batch_size"] * tr["gradient_accumulation_steps"])))
    total_train_steps = steps_per_epoch * tr["num_train_epochs"]
    warmup_steps = round(tr["warmup_ratio"] * total_train_steps)
    logger.info("Steps/epoch: %d | warmup_steps: %d", steps_per_epoch, warmup_steps)

    kwargs: Dict[str, Any] = dict(
        output_dir=str(output_dir),
        report_to=[],  # no w&b/tensorboard; we dump our own loss curves
        remove_unused_columns=False,
        num_train_epochs=tr["num_train_epochs"],
        per_device_train_batch_size=tr["batch_size"],
        gradient_accumulation_steps=tr["gradient_accumulation_steps"],
        learning_rate=tr["learning_rate"],
        lr_scheduler_type=tr["lr_scheduler_type"],
        warmup_steps=warmup_steps,
        weight_decay=tr["weight_decay"],
        max_grad_norm=tr["max_grad_norm"],
        bf16=tr["bf16"],
        fp16=tr["fp16"],
        gradient_checkpointing=tr["gradient_checkpointing"],
        logging_steps=tr["logging_steps"],
        eval_strategy=tr["eval_strategy"],
        save_strategy=tr["save_strategy"],
        save_total_limit=tr["save_total_limit"],
        load_best_model_at_end=tr["load_best_model_at_end"],
        metric_for_best_model=tr["metric_for_best_model"],
        seed=tr["seed"],
    )
    if tr["eval_strategy"] == "steps":
        kwargs["eval_steps"] = tr["eval_steps"]
        if kwargs.get("save_strategy") == "steps":
            kwargs["save_steps"] = tr["eval_steps"]

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**kwargs),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
    )

    logger.info("Starting fine-tune...")
    trainer.train()

    # final weights -> models/fine_tuned/final
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("Saved final weights + tokenizer -> %s", final_dir)

    dump_loss_curves(trainer, cfg)
    logger.info("Done.")


if __name__ == "__main__":
    main()