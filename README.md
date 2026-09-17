<h1 align="center">TrustReviewer</h1>

<p align="center">
  <b>When AI Reviews Train AI Reviewers: Scientific-Judgment Collapse and Mitigation</b>
</p>

<p align="center">
  <a href="">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg" alt="Paper">
  </a>
  <a href="https://github.com/hosytuyen/TrustReviewer">
    <img src="https://img.shields.io/badge/Code-GitHub-181717.svg" alt="Code">
  </a>
  <a href="https://huggingface.co/collections/hosytuyen/trustreviewer">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-ffcc4d.svg" alt="Dataset/Model">
  </a>
  <a href="https://hosytuyen.github.io/projects/TrustReviewer/">
    <img src="https://img.shields.io/badge/Project-Page-2ea44f.svg" alt="Project">
  </a>
</p>

Abstract: Large language models (LLMs) increasingly participate in scientific evaluation, both as automated reviewers and as assistants to human reviewers. As model-generated reviews enter public data and future training corpora, AI peer review can become recursive: later reviewers learn from judgments produced by earlier models. We study one step of this feedback loop in a controlled setting. Starting from Llama 3.1 8B, we first fine-tune a reviewer on official ICLR reviews from 2018--2023 and then train four successor models on ICLR 2024 data with systematically varied mixtures of official and model-generated reviews. Our study shows that introducing synthetic reviews compresses rating distributions and reduces both same-paper and corpus-level semantic diversity. We call this pattern $\textbf{scientific-judgment collapse}$.

To mitigate this failure mode, we introduce $\textbf{TrustReviewer}$, an open-source LLM-based system for generating peer reviews of AI and machine learning papers. TrustReviewer intervenes at two complementary stages. For training-time prevention, we train the core reviewer in a single stage on a curated corpus designed to reduce low-quality and semantically degenerate supervision. For test-time correction, paired activation steering aims to further mitigate residual tendencies toward collapsed judgments without further training or additional expert annotation. Together, these results characterize a concrete risk of recursive reviewer training and provide practical interventions for preserving judgment diversity and improving recommendation alignment in AI-assisted scientific evaluation.

## News

- **Sep 18, 2026**: Released the data/model on HF, and evaluation code!


## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download Model and Data


```bash
pip install -U huggingface_hub

hf download hosytuyen/TrustReviewer \
  --local-dir ./model

# Fixed held-out split and curated training corpus.
hf download hosytuyen/TrustReviewer \
  --repo-type dataset \
  --local-dir ./data
```

This creates `model/`, `data/eval.json`, and `data/training.json`. 

## Generate Reviews

```bash
python evaluation/generate_reviews.py \
  --papers-jsonl /path/to/held_out_papers.jsonl \
  --output-dir outputs/eval
```

Both generation commands use the included ICLR 2025 review form by default. Prompt assets are in `prompts/` and can be overridden without editing code:

```bash
python evaluation/generate_reviews.py \
  --papers-jsonl /path/to/held_out_papers.jsonl \
  --output-dir outputs/custom_prompt \
  --system-prompt-path /path/to/system.md \
  --review-template-path /path/to/review_fields.md \
  --user-prompt-template-path /path/to/user.md \
  --user-phrasing "Review this submission critically and constructively."
```

The system template must contain `{review_fields}`. The user template must contain `{user_phrasing}` and `{paper_text}`.

## Generate Steered Reviews

The released steering configuration applies the vector at layer 31 with coefficient 0.15.

```bash
python evaluation/generate_steered_reviews.py \
  --papers-jsonl /path/to/held_out_papers.jsonl \
  --output-dir outputs/steered_first_run
```

The minimal wrapper in `evaluation/repe/` is copied from the MIT-licensed
[`representation-engineering`](https://github.com/andyzoujm/representation-engineering) project.
The vector was estimated using the TrustReviewer's review and official review.

## Recommendation Matching and MAD
Following, [OpenReviewer](https://arxiv.org/abs/2412.11948), we use Recommendation Matching and MAD for the evaluation.


```bash
python evaluation/recommendation_matching.py \
  --review-dir outputs/eval \
  --human-review-dir /path/to/human_review
```

The report contains:

- `exact_match_rate`: whether a generated rating exactly matches at least one official review rating.
- `mean_abs_distance_to_human_mean`: MAD between the generated rating and the mean valid official rating for the paper.

## Training TrustReviewer and Construct Steering Vector (Optional)

### Training of TrustReviewer
Prepare shard dataset:
```bash
python training/shard_dataset.py \
    --input data/training.json \
    --output-dir data/training_sharded
```
Then, we finetune Meta-Llama-3.1-8B-Instruct on our curated data (see our [`training configuration`](./training/training_cfg.yaml)) using [`LLaMAFactory`](https://github.com/hiyouga/LlamaFactory)

**Construct Steering Vector.** We sample 5,000 input papers from training corpus, use TrustReviewer's generated reviews as negative data and their official reviews as positive data. The steering vector is constructed using [`representation-engineering`](https://github.com/andyzoujm/representation-engineering)
