import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shard a large LlamaFactory JSON dataset into smaller JSON files.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples-per-shard", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.examples_per_shard <= 0:
        raise ValueError("--examples-per-shard must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as handle:
        data = json.load(handle)

    for shard_index, start in enumerate(range(0, len(data), args.examples_per_shard)):
        shard = data[start : start + args.examples_per_shard]
        shard_path = args.output_dir / f"part_{shard_index:04d}.json"
        shard_path.write_text(json.dumps(shard, indent=2), encoding="utf-8")
        print(f"Wrote {shard_path} with {len(shard)} examples.")


if __name__ == "__main__":
    main()
