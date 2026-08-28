import gc
import json
import os
import re
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from izzyviz import cluster_attention_heads, visualize_attention_overview_fast


PROJECT_DIR = Path("/home/cuizhouying/IzzyViz")
OUTPUT_ROOT = PROJECT_DIR / "workflow" / "rag_cluster" / "multi_model_single_prompt_cluster"

MODEL_IDS = {
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B",
    "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-1.5B-Math": "Qwen/Qwen2.5-Math-1.5B",
    "Qwen2.5-1.5B-Math-Instruct": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "Qwen2.5-1.5B-Coder": "Qwen/Qwen2.5-Coder-1.5B",
    "DeepSeek-R1-Distill-Qwen-1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
}

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


def build_prompt():
    context_text = f"Context: {BASE_CONTEXT}\n"
    question_text = f"{context_text}Question: {QUESTION}\n"
    answer_prefix = "Answer:"
    return question_text + answer_prefix + f" {ANSWER}", question_text, answer_prefix


def get_model_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def run_model(model_label, model_id):
    output_dir = OUTPUT_ROOT / safe_name(model_label)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {model_label}: {model_id} ===", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=get_model_dtype(),
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()

    prompt, question_text, answer_prefix = build_prompt()
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

    x_labels = tokens[:question_end]
    y_labels = tokens[answer_start:]

    overview_merge_png_path = output_dir / "overview_merge_tokens.png"
    overview_merge_svg_path = output_dir / "overview_merge_tokens.svg"
    for overview_path in [overview_merge_png_path, overview_merge_svg_path]:
        visualize_attention_overview_fast(
            outputs.attentions,
            query_slice=(answer_start, None),
            key_slice=(0, question_end),
            title=f"{model_label}: answer -> context/question attention (merged virtual tokens)",
            save_path=overview_path,
            shared_color_scale=True,
            shared_cbar=True,
            shared_cbar_label="Attention Score",
            merge_virtual_tokens=True,
            overview_top_n=3,
            show_merge_token_labels=True,
            merge_token_label_mode="index",
            merge_token_important_label_mode="index",
            merge_token_label_fontsize=3,
            x_labels=x_labels,
            y_labels=y_labels,
            wspace=0.24,
            hspace=0.26,
            close_after_save=True,
        )

    result = cluster_attention_heads(
        outputs.attentions,
        query_slice=(answer_start, None),
        key_slice=(0, question_end),
        tokens=tokens,
        n_clusters=5,
        ignore_first_n=1,
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
        max_detail_plots=5,
        output_dir=output_dir,
        run_name=f"{model_label}_single_prompt_cluster",
        overview_renderer="fast",
        plot_overview=True,
        plot_overview_no_merge=True,
        plot_overview_merge=True,
        overview_kwargs={
            "shared_cbar": True,
            "shared_cbar_label": "Attention Score",
            "show_merge_token_labels": True,
            "merge_token_labels_representatives_only": False,
            "merge_token_label_mode": "index",
            "merge_token_important_label_mode": "index",
            "merge_token_label_fontsize": 3,
            "merge_token_highlight_color": "#f8bbd0",
            "merge_token_highlight_edgecolor": "#F06292",
            "merge_token_highlight_alpha": 0.7,
            "wspace": 0.24,
            "hspace": 0.26,
        },
        plot_pca=True,
        plot_detail_heatmaps=True,
        detail_top_n=20,
        detail_merge_virtual_tokens=True,
        detail_kwargs={
            "xlabel": "Context + Question",
            "ylabel": "Answer",
            "length_threshold": 64,
            "if_interval": False,
            "if_top_cells": True,
            "show_scores_in_enlarged_cells": False,
            "lean_more": True,
        },
    )

    npz_path = output_dir / "head_features.npz"
    head_infos_array = np.asarray(result["head_infos"], dtype=int)
    num_layers = int(head_infos_array[:, 0].max()) + 1
    num_heads = int(head_infos_array[:, 1].max()) + 1
    features_array = np.asarray(result["features"], dtype=float)
    features_scaled_array = np.asarray(result["features_scaled"], dtype=float)
    np.savez_compressed(
        npz_path,
        features=features_array,
        features_scaled=features_scaled_array,
        features_layer_head=features_array.reshape(num_layers, num_heads, -1),
        features_scaled_layer_head=features_scaled_array.reshape(num_layers, num_heads, -1),
        feature_names=np.asarray(result["feature_names"], dtype=object),
        head_infos=head_infos_array,
    )

    metadata = {
        "model_label": model_label,
        "model_id": model_id,
        "seq_len": int(inputs["input_ids"].shape[1]),
        "question_end": int(question_end),
        "answer_start": int(answer_start),
        "ignore_first_n": 1,
        "feature_path": str(npz_path),
        "output_paths": result["output_paths"],
    }
    with open(output_dir / "model_run_context.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    del outputs, result, model, tokenizer, inputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    plt.close("all")
    return metadata


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for model_label, model_id in MODEL_IDS.items():
        try:
            summaries.append(run_model(model_label, model_id))
        except Exception as exc:
            print(f"ERROR {model_label}: {repr(exc)}", flush=True)
            summaries.append({"model_label": model_label, "model_id": model_id, "error": repr(exc)})
    with open(OUTPUT_ROOT / "multi_model_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print("Saved", OUTPUT_ROOT / "multi_model_summary.json", flush=True)


if __name__ == "__main__":
    main()
