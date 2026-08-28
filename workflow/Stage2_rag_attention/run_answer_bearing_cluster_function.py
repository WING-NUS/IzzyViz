import gc
import json
import os
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from izzyviz import cluster_attention_heads


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_DIR / "workflow" / "rag_cluster" / "rag_contexts_answer_bearing"
OUTPUT_ROOT = DATASET_DIR / "exp_function"

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
N_CLUSTERS = 5
MAX_DETAIL_PLOTS = 5
IGNORE_FIRST_N = 1
FORCE_RERUN = os.environ.get("IZZYVIZ_FORCE_RERUN", "0") == "1"
HF_LOCAL_FILES_ONLY = os.environ.get("IZZYVIZ_HF_LOCAL_FILES_ONLY", "0") == "1"
CONTEXT_IDS = [
    value.strip()
    for value in os.environ.get("IZZYVIZ_CONTEXT_IDS", "").split(",")
    if value.strip()
]
MAX_CONTEXTS = os.environ.get("IZZYVIZ_MAX_CONTEXTS")
MAX_CONTEXTS = int(MAX_CONTEXTS) if MAX_CONTEXTS else None

BASE_CONTEXT = (
    "The Rev. John J. Cavanaugh, C.S.C. served as president from 1946 to 1952. "
    "Cavanaugh's legacy at Notre Dame in the post-war years was devoted to raising academic standards "
    "and reshaping the university administration to suit it to an enlarged educational mission and an expanded "
    "student body and stressing advanced studies and research at a time when Notre Dame quadrupled in student census, "
    "undergraduate enrollment increased by more than half, and graduate student enrollment grew fivefold. "
    "Cavanaugh also established the Lobund Institute for Animal Studies and Notre Dame's Medieval Institute. "
    "Cavanaugh also presided over the construction of the Nieuwland Science Hall, Fisher Hall, and the Morris Inn, "
    "as well as the Hall of Liberal Arts (now O'Shaughnessy Hall), made possible by a donation from I.A. O'Shaughnessy, "
    "at the time the largest ever made to an American Catholic university. "
    "Cavanaugh also established a system of advisory councils at the university, which continue today and are vital "
    "to the university's governance and development."
)
QUESTION = "Which institute involving animal life did Cavanaugh create at Notre Dame?"
ANSWER = "Lobund Institute for Animal Studies"


def safe_name(value):
    value = str(value)
    value = re.sub(r"[^a-zA-Z0-9_\-]+", "_", value)
    return value.strip("_") or "NA"


def build_prompt(context, question=QUESTION, answer=ANSWER, rag_context=None):
    if rag_context is None:
        context_text = f"Context: {context}\n"
    else:
        context_text = f"RAG Context: {rag_context}\nContext: {context}\n"
    question_text = f"{context_text}Question: {question}\n"
    answer_prefix = "Answer:"
    prompt = question_text + answer_prefix + f" {answer}"
    return prompt, question_text, answer_prefix


def get_attention_slices(prompt, question_text, answer_prefix, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    first_device = next(model.parameters()).device
    inputs = {key: value.to(first_device) for key, value in inputs.items()}
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    question_end = len(tokenizer(question_text, add_special_tokens=False)["input_ids"])
    answer_start = len(tokenizer(question_text + answer_prefix, add_special_tokens=False)["input_ids"])

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            return_dict=True,
            use_cache=False,
        )

    return outputs.attentions, tokens, question_end, answer_start, int(inputs["input_ids"].shape[1])


def get_model_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def is_complete(output_dir):
    output_dir = Path(output_dir)
    if not (output_dir / "run_summary.json").is_file():
        return False
    if not (output_dir / "pca_scatter.png").is_file():
        return False
    if not (output_dir / "overview.png").is_file():
        return False
    if not (output_dir / "overview_no_merge.png").is_file():
        return False
    if not (output_dir / "overview_merge_tokens.png").is_file():
        return False
    return len(list(output_dir.glob("cluster_*_L*_H*.pdf"))) >= MAX_DETAIL_PLOTS


def clean_previous_outputs(output_dir):
    output_dir = Path(output_dir)
    patterns = [
        "cluster_*_L*_H*.pdf",
        "pca_scatter.png",
        "overview.png",
        "overview_no_merge.png",
        "overview_merge_tokens.png",
        "run_summary.json",
        "function_run_context.json",
    ]
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def run_one(ctx_id, ctx_item, model, tokenizer):
    folder_name = (
        f"{ctx_id}_"
        f"{ctx_item.get('length', 'NA')}_"
        f"{safe_name(ctx_item.get('relatedness', 'NA'))}"
    )
    output_dir = OUTPUT_ROOT / folder_name

    if not FORCE_RERUN and is_complete(output_dir):
        print(f"SKIP complete: {ctx_id}", flush=True)
        return "skipped", output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    clean_previous_outputs(output_dir)
    with open(output_dir / f"{ctx_id}.json", "w", encoding="utf-8") as f:
        json.dump({ctx_id: ctx_item}, f, ensure_ascii=False, indent=2)

    prompt, question_text, answer_prefix = build_prompt(
        context=BASE_CONTEXT,
        question=QUESTION,
        answer=ANSWER,
        rag_context=ctx_item["text"],
    )
    attentions, tokens, question_end, answer_start, seq_len = get_attention_slices(
        prompt=prompt,
        question_text=question_text,
        answer_prefix=answer_prefix,
        model=model,
        tokenizer=tokenizer,
    )

    print(
        f"RUN {ctx_id}: seq_len={seq_len} answer_start={answer_start} question_end={question_end}",
        flush=True,
    )

    result = cluster_attention_heads(
        attentions,
        query_slice=(answer_start, None),
        key_slice=(0, question_end),
        tokens=tokens,
        n_clusters=N_CLUSTERS,
        ignore_first_n=IGNORE_FIRST_N,
        metrics=[
            "concentration",
            "entropy",
            "variance",
            "threshold_mass",
            "top_percent_mass",
            "low_range",
            "long_range",
        ],
        metric_params={
            "threshold_mass": {"threshold": 0.05},
            "top_percent_mass": {"top_percent": 0.05},
            "low_range": {"max_distance": 5},
            "long_range": {"min_distance": 20},
        },
        max_detail_plots=MAX_DETAIL_PLOTS,
        output_dir=output_dir,
        run_name=f"{ctx_id}_cluster_attention_heads",
        overview_renderer="fast",
        overview_merge_virtual_tokens=False,
        plot_overview_no_merge=True,
        plot_overview_merge=True,
        overview_kwargs={
            "shared_cbar": True,
            "shared_cbar_label": "Attention Score",
            "show_merge_token_labels": True,
            "merge_token_labels_representatives_only": False,
            "merge_token_label_mode": "index",
            "merge_token_important_label_mode": "index_token",
            "merge_token_label_fontsize": 3,
            "merge_token_highlight_color": "#f8bbd0",
            "merge_token_highlight_edgecolor": "#F06292",
            "merge_token_highlight_alpha": 0.7,
            "wspace": 0.24,
            "hspace": 0.26,
        },
        plot_overview=True,
        plot_pca=True,
        plot_detail_heatmaps=True,
        detail_top_n=20,
        detail_merge_virtual_tokens=True,
        detail_kwargs={
            "xlabel": "RAG Context + Context + Question",
            "ylabel": "Answer",
            "length_threshold": 64,
            "if_interval": False,
            "if_top_cells": True,
            "show_scores_in_enlarged_cells": False,
            "lean_more": True,
        },
    )

    metadata = {
        "dataset_dir": str(DATASET_DIR),
        "ctx_id": ctx_id,
        "length": ctx_item.get("length"),
        "relatedness": ctx_item.get("relatedness"),
        "seq_len": seq_len,
        "question_end": question_end,
        "answer_start": answer_start,
        "ignore_first_n": IGNORE_FIRST_N,
        "output_paths": result["output_paths"],
    }
    with open(output_dir / "function_run_context.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    del attentions, result
    return "ran", output_dir


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    contexts_path = DATASET_DIR / "external_contexts.json"
    with open(contexts_path, "r", encoding="utf-8") as f:
        contexts = json.load(f)

    print(f"Loading model: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=False,
        local_files_only=HF_LOCAL_FILES_ONLY,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=get_model_dtype(),
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=False,
        local_files_only=HF_LOCAL_FILES_ONLY,
    )
    model.eval()
    print("model device:", next(model.parameters()).device, flush=True)
    print(f"Loaded {len(contexts)} contexts from {contexts_path}", flush=True)
    print(f"Output root: {OUTPUT_ROOT}", flush=True)

    counts = {"ran": 0, "skipped": 0, "error": 0}
    items = list(contexts.items())
    if CONTEXT_IDS:
        requested = set(CONTEXT_IDS)
        items = [(ctx_id, ctx_item) for ctx_id, ctx_item in items if ctx_id in requested]
        missing = sorted(requested - {ctx_id for ctx_id, _ in items})
        if missing:
            raise ValueError(f"Requested context ids not found: {missing}")
    if MAX_CONTEXTS is not None:
        items = items[:MAX_CONTEXTS]

    for index, (ctx_id, ctx_item) in enumerate(items, start=1):
        print("=" * 80, flush=True)
        print(f"[{index}/{len(items)}] {ctx_id}", flush=True)
        try:
            status, output_dir = run_one(ctx_id, ctx_item, model, tokenizer)
            counts[status] += 1
            print(f"{status.upper()}: {output_dir}", flush=True)
        except Exception as exc:
            counts["error"] += 1
            print(f"ERROR {ctx_id}: {repr(exc)}", flush=True)
        finally:
            plt.close("all")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(0.2)

    summary = {
        "dataset_dir": str(DATASET_DIR),
        "output_root": str(OUTPUT_ROOT),
        "model_name": MODEL_NAME,
        "n_contexts": len(contexts),
        "n_contexts_run": len(items),
        "context_ids": [ctx_id for ctx_id, _ in items],
        "ignore_first_n": IGNORE_FIRST_N,
        "counts": counts,
    }
    with open(OUTPUT_ROOT / "exp_function_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("SUMMARY:", summary, flush=True)


if __name__ == "__main__":
    main()
