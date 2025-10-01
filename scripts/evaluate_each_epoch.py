import os
import argparse

import torch
import json
import shutil
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    get_tfidf_vectorizer,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from util import nethook
from util.globals import *

ds_map = {
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
}

def summarize_eval_results(result_dir: Path):
    result_groups = defaultdict(list)
    for path in sorted(result_dir.glob("*_edits-case_*.json")):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            num_edits = data.get("num_edits", 0)
            result_groups[num_edits].append(data)
        except Exception as e:
            print(f"Skipping {path.name} due to error: {e}")

    summary_data = {}

    for edit_count in sorted(result_groups):
        entries = result_groups[edit_count]
        print(f"\n==> Summary for edit_{edit_count} (n={len(entries)})")

        metrics = defaultdict(list)
        for entry in entries:
            for prefix in ["pre", "post"]:
                if prefix not in entry or entry[prefix] is None:
                    continue

                # ===== 分类类指标 =====
                for key in ["rewrite", "paraphrase", "neighborhood", "singlehop", "multihop"]:
                    full_key = f"{key}_prompts_correct"
                    if full_key in entry[prefix]:
                        metrics[f"{prefix}_{key}_acc"].append(np.mean(entry[prefix][full_key]))

                for key in ["rewrite", "paraphrase"]:
                    full_key = f"{key}_prompts_probs"
                    if full_key in entry[prefix]:
                        vals = entry[prefix][full_key]
                        metrics[f"{prefix}_{key}_success"].append(
                            np.mean([x["target_true"] > x["target_new"] for x in vals])
                        )
                        metrics[f"{prefix}_{key}_diff"].append(
                            np.mean([np.exp(-x["target_new"]) - np.exp(-x["target_true"]) for x in vals])
                        )

                key = "neighborhood_prompts_probs"
                if key in entry[prefix]:
                    vals = entry[prefix][key]
                    metrics[f"{prefix}_neighborhood_success"].append(
                        np.mean([x["target_true"] < x["target_new"] for x in vals])
                    )
                    metrics[f"{prefix}_neighborhood_diff"].append(
                        np.mean([np.exp(-x["target_true"]) - np.exp(-x["target_new"]) for x in vals])
                    )

                # ===== 生成类指标 =====
                for gen_key in ["ngram_entropy", "reference_score", "essence_score"]:
                    if gen_key in entry[prefix]:
                        metrics[f"{prefix}_{gen_key}"].append(entry[prefix][gen_key])

        # ===== 汇总输出 =====
        edit_summary = {}
        for k, v in metrics.items():
            mean = np.mean(v)
            std = np.std(v)

            # 除 essence_score 外，其余都 ×100
            if "essence_score" not in k:
                mean, std = mean * 100, std * 100

            print(f"{k:<32}: {mean:.2f} ± {std:.2f}")
            edit_summary[k] = {"mean": mean, "std": std}

        # harmonic scores
        for prefix in ["pre", "post"]:
            k_eff = f"{prefix}_rewrite_success"
            k_gen = f"{prefix}_paraphrase_success"
            k_spec = f"{prefix}_neighborhood_success"
            if all(k in metrics for k in [k_eff, k_gen, k_spec]):
                h = np.mean([np.mean(metrics[k]) for k in [k_eff, k_gen, k_spec]]) * 100
                print(f"{prefix}_harmonic_score{'':<17}: {h:.2f}")
                edit_summary[f"{prefix}_harmonic_score"] = h

        # reasoning scores
        for prefix in ["pre", "post"]:
            reasoning_keys = [f"{prefix}_{k}_acc" for k in ["singlehop", "multihop"]]
            if all(k in metrics for k in reasoning_keys):
                h_reasoning = np.mean([np.mean(metrics[k]) for k in reasoning_keys]) * 100
                print(f"{prefix}_reasoning_harmonic_score{'':<6}: {h_reasoning:.2f}")
                edit_summary[f"{prefix}_reasoning_harmonic_score"] = h_reasoning

        summary_data[f"edit_{edit_count}"] = edit_summary

    return summary_data

def evaluate_saved_weights(model_path, weights_dir, ds_name, dataset_size_limit, generation_test_interval):
    summary_log = {}

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.pad_token = tok.eos_token

    print("Loading dataset...")
    ds_class, eval_fn = ds_map[ds_name]
    dataset = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)

    snips = AttributeSnippets(DATA_DIR)
    vec = get_tfidf_vectorizer(DATA_DIR)

    skip_epochs = {'0'}

    weight_files = sorted(
        [f for f in Path(weights_dir).glob("edit_*.pth")
         if f.stem.split("_")[1] not in skip_epochs],
        key=lambda x: int(x.stem.split("_")[1])
    )

    result_dir = Path(weights_dir) / "eval_results"
    base_result_dir = Path(weights_dir) / "base_results"
    base_result_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否需要跑 base
    base_missing = any(
        not (base_result_dir / f"base_edits-case_{record['case_id']}.json").exists()
        for record in dataset
    )

    if base_missing:
        print("\n==> Base results missing, running original (unmodified) model evaluation...")
        for record in tqdm(dataset):
            out_file = base_result_dir / f"base_edits-case_{record['case_id']}.json"
            gen_test_vars = [snips, vec] if record["case_id"] % generation_test_interval == 0 else [None, None]
            result = eval_fn(model, tok, record, *gen_test_vars)
            metrics = {
                "case_id": record["case_id"],
                "requested_rewrite": record["requested_rewrite"],
                "num_edits": 0,
                "post": result,
            }
            with open(out_file, "w") as f:
                json.dump(metrics, f, indent=2)
    else:
        print("\n✅ Base results detected, skipping original model evaluation.")

    summary_log["base"] = summarize_eval_results(base_result_dir)

    for weight_file in weight_files:

        print(f"\n==> Evaluating {weight_file.name}")
        checkpoint = torch.load(weight_file, map_location="cpu")
        weights = checkpoint["weight"]

        with torch.no_grad():
            for name, param in weights.items():
                nethook.get_parameter(model, name)[...] = param.cuda()

        edit_id = int(weight_file.stem.split("_")[1])

        if result_dir.exists():
            shutil.rmtree(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)

        for record in tqdm(dataset[:edit_id]):
            case_id = record["case_id"]
            out_file = result_dir / f"{edit_id}_edits-case_{case_id}.json"
            gen_test_vars = [snips, vec] if case_id % generation_test_interval == 0 else [None, None]

            base_file = base_result_dir / f"base_edits-case_{case_id}.json"
            if base_file.exists():
                with open(base_file, "r") as f:
                    base_data = json.load(f)
                    pre_result = base_data["post"]
            else:
                pre_result = None

            post_result = eval_fn(model, tok, record, *gen_test_vars)

            metrics = {
                "case_id": case_id,
                "requested_rewrite": record["requested_rewrite"],
                "num_edits": edit_id,
                "pre": pre_result,
                "post": post_result,
            }
            with open(out_file, "w") as f:
                json.dump(metrics, f, indent=2)

        summary_log[f"edit_{edit_id}"] = summarize_eval_results(result_dir)

    summary_output_dir = Path(weights_dir) / "summary"
    summary_output_dir.mkdir(exist_ok=True, parents=True)
    with open(summary_output_dir / "summary.json", "w") as f:
        json.dump(summary_log, f, indent=2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/cache1/chtan/large_models/Llama-3/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--weight_folder", type=str, default='./Edited_Weight/MEMIT/Llama3-8B/mcf_weight_data_batch_100_0.8_0.8')

    parser.add_argument("--ds_name", choices=["cf", "mcf", "zsre"], default="mcf")
    parser.add_argument("--dataset_size_limit", type=int, default=10000)
    parser.add_argument("--generation_test_interval", type=int, default=100)
    args = parser.parse_args()

    evaluate_saved_weights(
        model_path=args.model_name,
        weights_dir=args.weight_folder,
        ds_name=args.ds_name,
        dataset_size_limit=args.dataset_size_limit,
        generation_test_interval=args.generation_test_interval,
    )
