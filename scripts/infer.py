"""Run single-image inference with a merged ReworkVLM checkpoint."""

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model ID or merged checkpoint directory")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", default=256, type=int)
    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": device}, attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(args.model)
    with Image.open(args.image) as image:
        content = [{"type": "image", "image": image.convert("RGB")}, {"type": "text", "text": args.prompt}]
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
    print(processor.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
