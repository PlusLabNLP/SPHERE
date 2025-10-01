import os

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from util.generate import generate_fast

from .compute_u import compute_u
from .compute_v import compute_v
from .rome_hparams import ROMEHyperParams
from torch.linalg import eigh
CONTEXT_TEMPLATES_CACHE = None



GLOBAL_EDIT_COUNT = 0

def apply_rome_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: List[Dict],
    hparams: ROMEHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    beta_hse=0,
    alpha=0,
    save_weights=True,
    cache_template: Optional[str] = None,
    dataset='None',
    batch_size=100,
    DATASET_CONFIG=None,

) -> Tuple[AutoModelForCausalLM, List[str]]:
    """
    Returns a model with the desired changes.

    :param copy: If true, will preserve the original model while creating a new one to edit.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """
    request = request[0]
    global GLOBAL_EDIT_COUNT
    GLOBAL_EDIT_COUNT += 1
    cur_edit_count = GLOBAL_EDIT_COUNT

    weights_copy = {}
    modified_weights = {}

    if save_weights:
        save_path = f'./Edited_Weight/ROME/{hparams.model_name.split("/")[-1]}/{dataset}_weight_data_batch_{batch_size}_{beta_hse}_{alpha}/'
        os.makedirs(save_path, exist_ok=True)

    if copy:
        model = deepcopy(model)

    deltas = execute_rome(model, tok, request, hparams)

    with torch.no_grad():
        for w_name, (delta_u, delta_v) in deltas.items():
            upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            # if return_orig_weights and w_name not in weights_copy:
            #     weights_copy[w_name] = w.detach().clone()
            if w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            if beta_hse > 0:
                upd_matrix_proj, P_soft, U = project_B_into_soft_dirs(w, upd_matrix.float(), energy_ratio=beta_hse, alpha = alpha)
                w[...] += upd_matrix_proj.float()
            else:
                w[...] += upd_matrix

            if save_weights:
                modified_weights[w_name] = w.detach().clone()

    print(f"New weights successfully inserted into {list(deltas.keys())}")


    dataset = dataset.lower()
    if save_weights and dataset in DATASET_CONFIG:
        config_ = DATASET_CONFIG[dataset]
        cur_step = min(cur_edit_count * batch_size, config_["total"])

        if cur_edit_count == 1:
            torch.save({"weight": weights_copy}, os.path.join(save_path, f'edit_0.pth'))
            torch.save({"weight": modified_weights}, os.path.join(save_path, f'edit_{cur_step}.pth'))
        elif cur_step in config_["save_steps"]:
            torch.save({"weight": modified_weights}, os.path.join(save_path, f'edit_{cur_step}.pth'))

        print(f"Original and modified weights saved to {save_path}")


    return model, weights_copy




def project_B_into_soft_dirs(A, B, energy_ratio=0.5, alpha=0.5):
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


def execute_rome(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the ROME update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    request = deepcopy(request)
    if request["target_new"]["str"][0] != " ":
        # Space required for correct tokenization
        request["target_new"]["str"] = " " + request["target_new"]["str"]

    if '{}' not in request['prompt']:
        assert request['subject'] in request['prompt'] or \
               print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")

        request['prompt'] = request['prompt'].replace(request['subject'], '{}')

    print(
        f"Executing ROME algorithm for the update: "
        f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']}]"
    )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Update loop: sequentially intervene at each specified layer
    deltas = {}
    for layer in sorted(hparams.layers):
        # Compute rank-1 update matrix
        left_vector: torch.Tensor = compute_u(
            model,
            tok,
            request,
            hparams,
            layer,
            get_context_templates(model, tok, hparams.context_template_length_params),
        )
        print("Left vector shape:", left_vector.shape)
        right_vector: torch.Tensor = compute_v(
            model,
            tok,
            request,
            hparams,
            layer,
            left_vector,
            get_context_templates(model, tok, hparams.context_template_length_params),
        )
        print("Right vector shape:", right_vector.shape)

        with torch.no_grad():
            # Determine correct transposition of delta matrix
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            # Update model weights and record desired changes in `delta` variable
            weights[weight_name][...] += upd_matrix
            deltas[weight_name] = (
                left_vector.detach(),
                right_vector.detach(),
            )

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by ROME does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok, length_params):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = ["{}"] + [
            x.replace("{", "").replace("}", "") + ". {}"
            for x in sum(
                (
                    generate_fast(
                        model,
                        tok,
                        ["The", "Therefore", "Because", "I", "You"],
                        n_gen_per_prompt=n_gen // 5,
                        max_out_len=length,
                    )
                    for length, n_gen in length_params
                ),
                [],
            )
        ]

        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
