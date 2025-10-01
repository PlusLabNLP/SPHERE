import os
import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from baselines.ft import FTHyperParams, apply_ft_to_model
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    get_tfidf_vectorizer,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from memit import MEMITHyperParams
from memit.compute_z import get_module_input_output_at_words, compute_z
from memit.memit_main import apply_memit_to_model, get_context_templates
from memit.memit_seq_main import apply_memit_seq_to_model
from memit.memit_rect_main import apply_memit_rect_to_model
from AlphaEdit import AlphaEditHyperParams
from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model, get_cov
from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook
from util.globals import *

ALG_DICT = {
    "AlphaEdit": (AlphaEditHyperParams, apply_AlphaEdit_to_model),
    "MEMIT_seq": (MEMITHyperParams, apply_memit_seq_to_model),
    "MEMIT_prune": (MEMITHyperParams, apply_memit_to_model),
    "MEMIT_rect": (MEMITHyperParams, apply_memit_rect_to_model),
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
}

DATASET_CONFIG = {
    "zsre": {
        "total": 19086,
        "save_steps": [5000, 15000]
    },
    "cf": {
        "total": 21919,
        "save_steps": [5000, 15000]
    },
    "mcf": {
        "total": 20877,
        "save_steps": [5000, 15000]
    },
}


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
    beta_hse: float = 0.0,
    alpha: float = 0.0,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]
    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist
    if (
        continue_from_run is None
        or not (run_dir := RESULTS_DIR / dir_name / continue_from_run).exists()
    ):
        continue_from_run = None
    if continue_from_run is None:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            # id_list = [
            #     int(str(x).split("_")[-1])
            #     for x in alg_dir.iterdir()
            #     if str(x).split("_")[-1].isnumeric()
            # ]
            id_list = [
                int(re.search(r"run_(\d+)", x.name).group(1))
                for x in alg_dir.iterdir()
                if re.match(r"run_\d+_", x.name)
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}_{ds_name}_{beta_hse}_{alpha}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")
    if "MEMIT" in alg_name:
    # Get run hyperparameters
        params_path = (
            run_dir / "params.json"
            if continue_from_run is not None
            else HPARAMS_DIR / "MEMIT" / hparams_fname
        )
    else:
        params_path = (
            run_dir / "params.json"
            if continue_from_run is not None
            else HPARAMS_DIR / alg_name / hparams_fname
        )
    hparams = params_class.from_json(params_path)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model
    if type(model_name) is str:
        print("Instantiating model")
        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path

    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    if num_edits > 1:
        assert ds_name != "cf", f"{ds_name} does not support multiple edits"

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)
    # Get cache templates
    cache_template = None
    if use_cache:
        if any(alg in alg_name for alg in ["MEMIT","AlphaEdit", "MEMIT_seq", "MEMIT_prune", "MEMIT_rect"]):
            cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_MEMIT"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
            )
        else:
            cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_{alg_name}"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
            )
        print(f"Will load cache from {cache_template}")
    if alg_name == "NSE":
        cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_{alg_name}"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        for record in ds:
            # Retrieve k/v pair if already stored in cache
            cache_fname = (
                Path(
                    str(cache_template).format(
                        hparams.layers[-1], hparams.clamp_norm_factor, record["case_id"]
                    )
                )
                if cache_template is not None
                else None
            )
            data_loaded = False
            if (
                cache_fname is not None  # Require cache template
                and cache_fname.exists()  # Cache file must exist
            ):
                continue
            # Compute k/v pair if not loaded from cache
            if not data_loaded:
                context_templates = get_context_templates(model, tok)
                cur_z = compute_z(
                    model,
                    tok,
                    {"case_id": record["case_id"], **record["requested_rewrite"]},
                    hparams,
                    hparams.layers[-1],
                    context_templates,
                )
                if cache_fname is not None:
                    cache_fname.parent.mkdir(exist_ok=True, parents=True)
                    np.savez(
                        cache_fname,
                        **{
                            "v_star": cur_z.detach().cpu().numpy(),
                        },
                    )
                    print(f"Cached k/v pair at {cache_fname}")
    if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "MEMIT_prune", "NSE"]):
        # Iterate through dataset
        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        if hparams.model_name == "gpt2-xl":
            cache_c = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
            if alg_name == "AlphaEdit":
                P = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
        elif hparams.model_name in ["EleutherAI_gpt-j-6B","Llama3-8B","phi-1.5", "Qwen2.5-7B-Instruct"]:
            cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
            if alg_name == "AlphaEdit":
                P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
        del W_out
    if alg_name == "AlphaEdit":
        for i, layer in enumerate(hparams.layers):
            P[i,:,:] = get_project(model,tok,layer,hparams)
        torch.save(P, "null_space_project.pt")

    rect_save_location = str(run_dir) + '/' + 'rect_eval/'
    os.makedirs(rect_save_location, exist_ok=True)
    cnt = 0
    for record_chunks in chunks(ds, num_edits):
        case_result_template = str(run_dir / "{}_edits-case_{}.json")
        print(f"=================================================================={cnt+1}_edit==================================================================")
        # Is the chunk already done?
        already_finished = True
        for record in record_chunks:
            if not Path(
                case_result_template.format(num_edits, record["case_id"])
            ).exists():
                already_finished = False
                break
        if already_finished:
            continue
        
        # Compute weight changes + record weights that changed
        case_ids = [record["case_id"] for record in record_chunks]
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT","AlphaEdit", "MEMIT_seq", "MEMIT_prune", "NSE"]) else dict()
        seq_args = dict(cache_c=cache_c) if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "NSE"]) else dict()
        nc_args = dict(P = P) if any(alg in alg_name for alg in ["AlphaEdit"]) else dict()

        start = time()
        if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "NSE"]):
            edited_model, cache_c = apply_algo(
                model,
                tok,
                [
                    {"case_id": record["case_id"], **rewrite_dict}
                    for record in record_chunks
                    for rewrite_dict in (
                        record["requested_rewrite"]
                        if isinstance(record["requested_rewrite"], list)
                        else [record["requested_rewrite"]]
                    )
                ],
                hparams,
                beta_hse=beta_hse,
                alpha = alpha,
                dataset = ds_name,
                batch_size = num_edits,
                DATASET_CONFIG = DATASET_CONFIG,
                **args_conserve_memory,
                **etc_args,
                **seq_args,
                **nc_args,
            )
        elif alg_name == "MEMIT_prune":
            # 保存当前 batch 的修改后的权重
            save_path = f"./Edited_Weight/MEMIT_prune/{hparams.model_name.split('/')[-1]}/{ds_name}_weight_data_batch_{num_edits}_{beta_hse}_{alpha}/"
            os.makedirs(save_path, exist_ok=True)

            if cnt == 0:
                edited_model, weights_copy = apply_algo(
                    model,
                    tok,
                    [
                        {"case_id": record["case_id"], **rewrite_dict}
                        for record in record_chunks
                        for rewrite_dict in (
                            record["requested_rewrite"]
                            if isinstance(record["requested_rewrite"], list)
                            else [record["requested_rewrite"]]
                        )
                    ],
                    hparams,
                    return_orig_weights=True,
                    save_weights = False,
                    beta_hse=0,
                    dataset = ds_name,
                    batch_size = num_edits,
                    DATASET_CONFIG = DATASET_CONFIG,
                    **args_conserve_memory,
                    **etc_args,

                )
                # torch.save({"weight": weights_copy}, os.path.join(save_path, f"edit_0.pth"))

            else:
                edited_model, _ = apply_algo(
                    model,
                    tok,
                    [
                        {"case_id": record["case_id"], **rewrite_dict}
                        for record in record_chunks
                        for rewrite_dict in (
                            record["requested_rewrite"]
                            if isinstance(record["requested_rewrite"], list)
                            else [record["requested_rewrite"]]
                        )
                    ],
                    hparams,
                    return_orig_weights=False,
                    beta_hse=0,
                    save_weights=False,
                    dataset = ds_name,
                    batch_size = num_edits,
                    DATASET_CONFIG = DATASET_CONFIG,
                    **args_conserve_memory,
                    **etc_args,
                )

            # Calculate upd_matrix for this batch
            upd_matrix = {}
            modified_weights = {}
            with torch.no_grad():
                for k, v in weights_copy.items():
                    current_weight = nethook.get_parameter(model, k)
                    delta = current_weight - v.to("cuda")

                    # SVD调整
                    _, S_orig, _ = torch.svd(v)
                    max_sigma = S_orig.max().item()

                    U_upd, S_upd, V_upd = torch.svd(delta)
                    adjusted_S = torch.where(
                        S_upd > max_sigma,
                        torch.log(S_upd) - torch.log(torch.tensor(max_sigma, device='cuda')) + max_sigma,
                        S_upd
                    )
                    adjusted_delta = torch.matmul(U_upd, torch.matmul(torch.diag(adjusted_S), V_upd.t()))

                    if beta_hse > 0:
                        adjusted_delta_proj, P_soft, U = project_B_into_soft_dirs(v.to("cuda"), adjusted_delta.float(),
                                                                              energy_ratio=beta_hse, alpha=alpha)
                        adjusted_delta_final = adjusted_delta_proj.float()
                    else:
                        adjusted_delta_final = adjusted_delta

                    upd_matrix[k] = adjusted_delta_final
                    modified_weights[k] = v.to("cuda") + adjusted_delta_final


            # torch.save({"weight": modified_weights}, os.path.join(save_path, f"edit_{(cnt + 1)*num_edits}.pth"))
            #
            # print(f"[Batch {cnt}] Saved adjusted weights to {save_path}")
            dataset = ds_name.lower()
            if dataset in DATASET_CONFIG:
                config_ = DATASET_CONFIG[dataset]
                cur_step = min((cnt + 1) * num_edits, config_["total"])

                if (cnt + 1) == 1:
                    torch.save({"weight": weights_copy}, os.path.join(save_path, f'edit_0.pth'))
                    torch.save({"weight": modified_weights}, os.path.join(save_path, f'edit_{cur_step}.pth'))
                elif cur_step in config_["save_steps"]:
                    torch.save({"weight": modified_weights}, os.path.join(save_path, f'edit_{cur_step}.pth'))
                elif beta_hse < 0:
                    torch.save({"weight": modified_weights}, os.path.join(save_path, f'edit_{cur_step}.pth'))

                print(f"Original and modified weights saved to {save_path}")



        else:
            edited_model, _ = apply_algo(
                model,
                tok,
                [
                    {"case_id": record["case_id"], **rewrite_dict}
                    for record in record_chunks
                    for rewrite_dict in (
                        record["requested_rewrite"]
                        if isinstance(record["requested_rewrite"], list)
                        else [record["requested_rewrite"]]
                    )
                ],
                hparams,
                return_orig_weights=False,
                beta_hse=beta_hse,
                alpha = alpha,
                dataset=ds_name,
                batch_size=num_edits,
                DATASET_CONFIG=DATASET_CONFIG,
                **args_conserve_memory,
                **etc_args,
            )
        exec_time = time() - start
        cnt+=1
        print("Execution took", exec_time)

    start = time()
    gen_test_vars = [snips, vec]
    for record in ds:
        out_file = Path(case_result_template.format(num_edits, record["case_id"]))
        if out_file.exists():
            print(f"Skipping {out_file}; already exists")
            continue
        metrics = {
            "case_id": record["case_id"],
            "grouped_case_ids": case_ids,
            "num_edits": num_edits,
            "requested_rewrite": record["requested_rewrite"],
            "time": exec_time,
            "post": ds_eval_method(
                edited_model,
                tok,
                record,
                *(
                    gen_test_vars
                    if record["case_id"] % generation_test_interval == 0
                    else [None, None]
                ),  # Only test generation every generation_test_interval cases
            ),
        }
        # Dump metrics in .json
        with open(out_file, "w") as f:
            json.dump(metrics, f, indent=1)

        print("Evaluation took", time() - start)

def get_project(model, tok, layer, hparams):
    force_recompute = False
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples
        if not force_recompute
        else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
    ).cpu()
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    print(len(small_singular_indices))
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


from torch.linalg import eigh

def project_B_into_soft_dirs(A, B, energy_ratio=0.5, alpha=0.8):
    A_hat = A / (A.norm(dim=1, keepdim=True) + 1e-8)
    C = (A_hat.T @ A_hat) / A_hat.size(0)
    eigvals, eigvecs = eigh(C)
    cumsum = torch.cumsum(eigvals.flip(0), 0)
    total  = eigvals.sum()
    r = (cumsum / total <= energy_ratio).sum().item() + 1
    U = eigvecs[:, -r:]
    P_soft = torch.eye(A.size(1), device=A.device) - alpha * (U @ U.T)
    B_proj = B @ P_soft.T
    return B_proj, P_soft, U


if __name__ == "__main__":
    import argparse
    import logging
    import sys

    # 初始化日志，仅记录到文件
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("evaluate.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    def exception_hook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = exception_hook


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["AlphaEdit","MEMIT_rect", "MEMIT_seq","MEMIT_prune", "MEMIT", "ROME", "FT"],
        default="MEMIT_prune",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        default="",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="Qwen2.5-7B.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre"],
        default="mcf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=100,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=100,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=10,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        default=False,
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--beta_hse",
        type=float,
        default=0.0,
        help="The main directions ratio we want to avoid",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="The projection strength on the main directions, set 1 for totally orthogonal projection, set 0 for do nothing (no projection)",
    )

    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()


    try:
        main(
            args.alg_name,
            args.model_name,
            args.hparams_fname,
            args.ds_name,
            args.dataset_size_limit,
            args.continue_from_run,
            args.skip_generation_tests,
            args.generation_test_interval,
            args.conserve_memory,
            dir_name=args.alg_name,
            num_edits=args.num_edits,
            use_cache=args.use_cache,
            beta_hse=args.beta_hse,
            alpha=args.alpha,
        )
    except Exception:
        logging.exception("❌ An unhandled exception occurred during program execution.")
        raise
