"""Phase 4 validation: load base model + tokenizer and run one dummy pass.

Usage:
    python scripts/verify_model_loader.py            # defaults from model_config.yaml
    python scripts/verify_model_loader.py --dtype bfloat16 --question "Why should I diversify my portfolio?"

Steps:
  1. read configs/model_config.yaml (base model, precision, generation defaults);
  2. load tokenizer + base LM via src/model/model_loader.py (artifacts cached in models/base);
  3. run a single dummy forward pass and report the logits shape;
  4. run one generate() call on a sample financial question and print the raw output.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from src.model import model_loader  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Phase 4 model loader.")
    parser.add_argument("--dtype", default=None, help="Override load dtype (auto/float16/bfloat16/float32).")
    parser.add_argument(
        "--question",
        default="What is the difference between a Roth IRA and a traditional IRA?",
        help="Sample financial question used for the dummy generate() call.",
    )
    args = parser.parse_args()

    cfg = model_loader.load_config()
    print(f"Config: {cfg['model']['base_model']!r} | "
          f"dtype={cfg['model']['precision']['load_dtype']!r} | "
          f"quant={cfg['model']['precision']['quantization']!r} | "
          f"device={cfg['model']['device']!r}")
    print(f"Generation defaults: max_new_tokens={cfg['generation']['max_new_tokens']} "
          f"temperature={cfg['generation']['temperature']} top_p={cfg['generation']['top_p']}\n")

    device = model_loader.resolve_device(cfg["model"]["device"])
    print(f"Resolved device: {device}")

    # 1) Tokenizer --------------------------------------------------------
    t0 = time.perf_counter()
    tokenizer = model_loader.load_tokenizer()
    print(f"[tokenizer] loaded {cfg['model']['base_model']!r} "
          f"in {time.perf_counter() - t0:.1f}s (vocab={len(tokenizer)}, "
          f"pad={tokenizer.pad_token!r}, chat_template={'yes' if tokenizer.chat_template else 'no'})")

    # 2) Model ------------------------------------------------------------
    t0 = time.perf_counter()
    model = model_loader.load_model(load_dtype=args.dtype)
    print(f"[model] loaded in {time.perf_counter() - t0:.1f}s")
    print(f"[model] {model_loader.get_model_info(model)}\n")

    # 3) Dummy forward pass ------------------------------------------------
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    print(f"[forward pass] input_ids={tuple(inputs['input_ids'].shape)} "
          f"-> logits={tuple(outputs.logits.shape)} (dummy forward OK)\n")

    # 4) One generate() call ------------------------------------------------
    t0 = time.perf_counter()
    raw = model_loader.generate(model, tokenizer, args.question, device=device)
    print(f"[generate] produced {len(raw.split())} tokens in {time.perf_counter() - t0:.1f}s")
    print("RAW OUTPUT\n---------")
    print(raw)
    print("---------")


if __name__ == "__main__":
    main()