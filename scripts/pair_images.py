"""Verify ReworkVLM image paths and optionally materialize training JSONL."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath
import tempfile

PREFIXES = {"mvtec_loco_anomaly_detection": "MVTec-LOCO", "VID-AD": "VID-AD"}


def rows(path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def resolve_image(name, roots):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or ":" in name or "\\" in name or len(path.parts) < 2:
        raise ValueError(f"Invalid image path: {name}")
    root = roots[path.parts[0]]
    local = root.joinpath(*path.parts[1:]).resolve()
    if not local.is_relative_to(root) or not local.is_file():
        raise FileNotFoundError(name)
    return local


def sha256(path):
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def verify_bundle(bundle, roots):
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    for relative, info in manifest["files"].items():
        if sha256(bundle / relative) != info["sha256"]:
            raise ValueError(f"Annotation checksum mismatch: {relative}")
    names_on_disk = {
        prefix: {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
        for prefix, root in roots.items()
    }
    records = {}
    counts = Counter()
    for row in rows(bundle / "image_index.jsonl"):
        name = row["image"]
        prefix, relative = name.split("/", 1)
        if relative not in names_on_disk[prefix]:
            raise FileNotFoundError(f"Image missing (matching case required): {name}")
        if name in records:
            raise ValueError(f"Duplicate index entry: {name}")
        if row["dataset"] != PREFIXES[prefix]:
            raise ValueError(f"Dataset mismatch: {name}")
        records[name] = resolve_image(name, roots)
        counts[row["dataset"]] += 1
    print("Matched image index:", dict(counts))
    return manifest, records


def pair_row(row, records, roots):
    if "images" in row:
        row["images"] = [str(records[name]) for name in row["images"]]
    if "annotation" in row:
        row["remote_image"] = str(records[row["image"]])
        annotation = row["annotation"]
        annotation["reference_images"] = [str(records[name]) for name in annotation.get("reference_images", [])]
        annotation["mask_paths"] = [str(resolve_image(name, roots)) for name in annotation.get("mask_paths", [])]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--loco-root", required=True, type=Path)
    parser.add_argument("--vid-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--file", help="Materialize only this release-relative JSONL")
    args = parser.parse_args()
    roots = {"mvtec_loco_anomaly_detection": args.loco_root.resolve(), "VID-AD": args.vid_root.resolve()}
    bundle = args.bundle.resolve()
    manifest, records = verify_bundle(bundle, roots)
    inputs = [name for name in manifest["files"] if name.endswith(".jsonl") and name != "image_index.jsonl"]
    if args.file:
        if not args.output or args.file not in inputs:
            parser.error("--file requires --output and a listed dataset JSONL")
        inputs = [args.file]
    if not args.output:
        return
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(bundle):
        parser.error("Choose a new output directory outside the dataset bundle")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".reworkvlm-") as temp:
        staging = Path(temp) / "paired"
        for name in inputs:
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8", newline="\n") as stream:
                for row in rows(bundle / name):
                    stream.write(json.dumps(pair_row(row, records, roots), ensure_ascii=False, separators=(",", ":")) + "\n")
        staging.rename(output)
    print(f"Materialized {len(inputs)} JSONL files in {output}")


if __name__ == "__main__":
    main()
