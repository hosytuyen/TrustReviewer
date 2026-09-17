"""Generate TrustReviewer reviews with the released steering vector."""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate_reviews import (
    DEFAULT_PAPER_IDS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_REVIEW_TEMPLATE_PATH,
    DEFAULT_SYSTEM_PROMPT_PATH,
    DEFAULT_USER_PHRASING,
    DEFAULT_USER_PROMPT_TEMPLATE_PATH,
    build_prompts,
    load_paper_ids,
    load_papers,
    valid_rating,
)
from repe.rep_control_reading_vec import WrappedReadingVecModel


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEERING_PATH = ROOT / "steering" / "steering_vector.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--paper-ids-path", type=Path, default=DEFAULT_PAPER_IDS_PATH)
    parser.add_argument("--system-prompt-path", type=Path, default=DEFAULT_SYSTEM_PROMPT_PATH)
    parser.add_argument("--review-template-path", type=Path, default=DEFAULT_REVIEW_TEMPLATE_PATH)
    parser.add_argument("--user-prompt-template-path", type=Path, default=DEFAULT_USER_PROMPT_TEMPLATE_PATH)
    parser.add_argument("--user-phrasing", default=DEFAULT_USER_PHRASING)
    parser.add_argument("--steering-vector-path", type=Path, default=DEFAULT_STEERING_PATH)
    parser.add_argument("--layer", type=int, default=31)
    parser.add_argument("--coefficient", type=float, default=0.15)
    parser.add_argument("--max-input-tokens", type=int, default=64000)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_vector(path: Path, layer: int, coefficient: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    vectors = torch.load(path, map_location="cpu", weights_only=True)
    if layer not in vectors and str(layer) not in vectors:
        raise ValueError(f"Layer {layer} is absent from {path}")
    vector = vectors.get(layer, vectors.get(str(layer)))
    return (vector.to(torch.float32) * coefficient).to(device=device, dtype=dtype)


def main() -> None:
    args = parse_args()
    if not args.steering_vector_path.is_file():
        raise FileNotFoundError(args.steering_vector_path)
    build_prompts(args, "")
    paper_ids = load_paper_ids(args.paper_ids_path)
    papers = load_papers(args.papers_jsonl)
    missing = [paper_id for paper_id in paper_ids if paper_id not in papers]
    if missing:
        raise ValueError(f"Missing {len(missing)} held-out papers, e.g. {missing[:5]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type == "cpu":
        model.to(device)
    model.eval()
    vector = load_vector(args.steering_vector_path, args.layer, args.coefficient, model.device, dtype)
    wrapped_model = WrappedReadingVecModel(model, tokenizer)
    wrapped_model.wrap_block([args.layer], block_name="decoder_block")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    completed = skipped = failed = 0
    for paper_id in tqdm(paper_ids, desc="Generating steered reviews"):
        if args.limit is not None and completed >= args.limit:
            break
        output_path = args.output_dir / paper_id / "review.md"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        system_prompt, user_prompt = build_prompts(args, papers[paper_id])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_tokens).to(model.device)
        try:
            wrapped_model.reset()
            wrapped_model.set_controller(
                [args.layer], {args.layer: vector}, block_name="decoder_block", token_pos="end", masks=1
            )
            with torch.inference_mode():
                generated = wrapped_model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )
            review = tokenizer.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            if not valid_rating(review):
                raise ValueError("Generated review lacks a strict 1--10 rating under ## Rating.")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(review + "\n", encoding="utf-8")
            completed += 1
        except Exception as exc:
            failed += 1
            print(f"Failed {paper_id}: {exc}")
        finally:
            wrapped_model.reset()

    print(f"Completed: {completed}; skipped: {skipped}; failed: {failed}")


if __name__ == "__main__":
    main()
