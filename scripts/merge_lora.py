"""Merge one ReworkVLM RL adapter into a local Qwen3-VL base snapshot."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true", help="Reload the export and generate from a synthetic image")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("The output directory already exists")

    import torch
    from PIL import Image
    from peft import PeftModel
    from safetensors import safe_open
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    options = dict(dtype=torch.bfloat16, device_map={"": args.device},
                   attn_implementation="sdpa", local_files_only=True)
    base = Qwen3VLForConditionalGeneration.from_pretrained(args.base, **options)
    processor = AutoProcessor.from_pretrained(args.base, local_files_only=True)
    adapter_weights = args.adapter / "adapter_model.safetensors"
    with safe_open(adapter_weights, framework="pt") as checkpoint:
        adapter_tensors = len(checkpoint.keys())
    adapted = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    active_tensors = sum(".lora_A." in name or ".lora_B." in name for name, _ in adapted.named_parameters())
    if active_tensors != adapter_tensors:
        raise ValueError(f"Adapter tensor count mismatch: {active_tensors} active / {adapter_tensors} on disk")
    merged = adapted.merge_and_unload(safe_merge=True)
    # Use the canonical identifier in metadata instead of the source machine path.
    merged.config._name_or_path = "Qwen/Qwen3-VL-4B-Instruct"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".merge-", dir=args.output.parent) as temporary:
        staging = Path(temporary) / "export"
        merged.save_pretrained(staging, safe_serialization=True, max_shard_size="4GB")
        processor.save_pretrained(staging)
        del adapted, base, merged, processor
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

        index = json.loads((staging / "model.safetensors.index.json").read_text())
        for name in sorted(set(index["weight_map"].values())):
            expected = {key for key, shard in index["weight_map"].items() if shard == name}
            with safe_open(staging / name, framework="pt") as shard:
                if set(shard.keys()) != expected:
                    raise ValueError(f"Exported tensors do not match the index in {name}")
        report = {
            "adapter_sha256": digest(adapter_weights),
            "adapter_tensors": adapter_tensors,
            "software": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "safetensors")},
            "files": {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
                      for path in sorted(staging.iterdir()) if path.is_file()},
        }
        if args.smoke:
            model = Qwen3VLForConditionalGeneration.from_pretrained(staging, **options)
            processor = AutoProcessor.from_pretrained(staging, local_files_only=True)
            messages = [{"role": "user", "content": [
                {"type": "image", "image": Image.new("RGB", (112, 112), "red")},
                {"type": "text", "text": "What color is the image? Answer in one word."},
            ]}]
            batch = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                  return_dict=True, return_tensors="pt").to(args.device)
            with torch.inference_mode():
                generated = model.generate(**batch, max_new_tokens=10, do_sample=False)
            answer = processor.decode(generated[0, batch.input_ids.shape[1]:], skip_special_tokens=True).strip()
            if not answer:
                raise ValueError("The exported model produced an empty response")
            report["smoke_answer"] = answer
        if args.output.exists():
            raise FileExistsError(args.output)
        staging.rename(args.output)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
