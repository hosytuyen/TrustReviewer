"""Compute recommendation exact match and MAD against official reviews."""

import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER_IDS_PATH = ROOT / "data" / "eval.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--human-review-dir", type=Path, required=True)
    parser.add_argument("--paper-ids-path", type=Path, default=DEFAULT_PAPER_IDS_PATH)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def load_paper_ids(path: Path) -> list[str]:
    return [row["paper_id"] for row in json.loads(path.read_text(encoding="utf-8"))]


def parse_rating(path: Path) -> int | None:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^## Rating:?\s*$\n+\s*([0-9]+)\b", text, flags=re.MULTILINE)
    if not match:
        return None
    rating = int(match.group(1))
    return rating if 1 <= rating <= 10 else None


def main() -> None:
    args = parse_args()
    output_path = args.output_path or args.review_dir / "recommendation_matching_report.json"
    rows = []
    exact_matches = 0
    total_abs_distance = 0.0

    for paper_id in tqdm(load_paper_ids(args.paper_ids_path), desc="Scoring reviews"):
        generated_path = args.review_dir / paper_id / "review.md"
        human_paths = sorted((args.human_review_dir / paper_id).glob("*.md"))
        generated_rating = parse_rating(generated_path) if generated_path.is_file() else None
        human_ratings = [rating for path in human_paths if (rating := parse_rating(path)) is not None]

        if generated_rating is None:
            rows.append({"paper_id": paper_id, "status": "skipped", "reason": "invalid_or_missing_generated_rating"})
            continue
        if not human_ratings:
            rows.append({"paper_id": paper_id, "status": "skipped", "reason": "no_valid_official_rating"})
            continue

        human_mean = sum(human_ratings) / len(human_ratings)
        exact_match = generated_rating in human_ratings
        abs_distance = abs(generated_rating - human_mean)
        exact_matches += int(exact_match)
        total_abs_distance += abs_distance
        rows.append(
            {
                "paper_id": paper_id,
                "status": "evaluated",
                "generated_rating": generated_rating,
                "official_ratings": human_ratings,
                "official_mean_rating": human_mean,
                "exact_match": exact_match,
                "abs_distance_to_official_mean": abs_distance,
            }
        )

    evaluated = [row for row in rows if row["status"] == "evaluated"]
    report = {
        "review_dir": str(args.review_dir),
        "human_review_dir": str(args.human_review_dir),
        "held_out_papers": len(rows),
        "evaluated_papers": len(evaluated),
        "skipped_papers": len(rows) - len(evaluated),
        "exact_match_rate": exact_matches / len(evaluated) if evaluated else None,
        "mean_abs_distance_to_official_mean": total_abs_distance / len(evaluated) if evaluated else None,
        "per_paper": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "per_paper"}, indent=2))


if __name__ == "__main__":
    main()
