"""Generate TrustReviewer reviews for the fixed held-out paper split."""

import argparse
import json
import random
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "model"
DEFAULT_PAPER_IDS_PATH = ROOT / "data" / "eval.json"
PROMPT_DIR = ROOT / "prompts"
DEFAULT_SYSTEM_PROMPT_PATH = PROMPT_DIR / "system_prompt.md"
DEFAULT_REVIEW_TEMPLATE_PATH = PROMPT_DIR / "iclr_2025.md"
DEFAULT_USER_PROMPT_TEMPLATE_PATH = PROMPT_DIR / "user_prompt.md"
DEFAULT_USER_PHRASING = "Please review the following paper according to the ICLR reviewing guidelines."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--paper-ids-path", type=Path, default=DEFAULT_PAPER_IDS_PATH)
    parser.add_argument(
        "--system-prompt-path",
        type=Path,
        default=DEFAULT_SYSTEM_PROMPT_PATH,
        help="Markdown template with a {review_fields} placeholder.",
    )
    parser.add_argument(
        "--review-template-path",
        type=Path,
        default=DEFAULT_REVIEW_TEMPLATE_PATH,
        help="Markdown description of the required review fields.",
    )
    parser.add_argument(
        "--user-prompt-template-path",
        type=Path,
        default=DEFAULT_USER_PROMPT_TEMPLATE_PATH,
        help="Markdown template with {user_phrasing} and {paper_text} placeholders.",
    )
    parser.add_argument(
        "--user-phrasing",
        default=DEFAULT_USER_PHRASING,
        help="Instruction inserted into the user prompt template.",
    )
    parser.add_argument("--max-input-tokens", type=int, default=64000)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_paper_ids(path: Path) -> list[str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [row["paper_id"] for row in rows]


def load_papers(path: Path) -> dict[str, str]:
    papers = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        paper_id = row.get("paper_id")
        paper_text = row.get("paper_text")
        if not isinstance(paper_id, str) or not isinstance(paper_text, str):
            raise ValueError(f"Line {line_number} must contain string paper_id and paper_text fields.")
        if paper_id in papers:
            raise ValueError(f"Duplicate paper_id in {path}: {paper_id}")
        papers[paper_id] = paper_text
    return papers


def valid_rating(review_text: str) -> bool:
    match = re.search(r"^## Rating:?\s*$\n+\s*([0-9]+)\b", review_text, flags=re.MULTILINE)
    return bool(match and 1 <= int(match.group(1)) <= 10)


def load_prompt(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8").strip()


def build_prompts(args: argparse.Namespace, paper_text: str) -> tuple[str, str]:
    system_prompt = load_prompt(args.system_prompt_path).format(
        review_fields=load_prompt(args.review_template_path)
    )
    user_prompt = load_prompt(args.user_prompt_template_path).format(
        user_phrasing=args.user_phrasing,
        paper_text=paper_text,
    )
    return system_prompt, user_prompt


def main() -> None:
    args = parse_args()
    if args.max_input_tokens <= 0 or args.max_new_tokens <= 0:
        raise ValueError("Token limits must be positive.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")
    # Validate templates before loading the model or starting a long generation job.
    build_prompts(args, "")

    paper_ids = load_paper_ids(args.paper_ids_path)
    papers = load_papers(args.papers_jsonl)
    missing = [paper_id for paper_id in paper_ids if paper_id not in papers]
    if missing:
        raise ValueError(f"Missing {len(missing)} held-out papers, e.g. {missing[:5]}")
    if args.shuffle:
        random.Random(args.seed).shuffle(paper_ids)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model.to(device)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    completed = skipped = failed = 0
    for paper_id in tqdm(paper_ids, desc="Generating reviews"):
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
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_tokens,
        ).to(model.device)
        try:
            with torch.inference_mode():
                generated = model.generate(
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

    print(f"Completed: {completed}; skipped: {skipped}; failed: {failed}")


if __name__ == "__main__":
    main()
