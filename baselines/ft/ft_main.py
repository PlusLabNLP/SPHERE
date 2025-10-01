from copy import deepcopy
from typing import Any, Dict, List, Tuple
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from util import nethook

from .ft_hparams import FTHyperParams
from torch.linalg import eigh

GLOBAL_EDIT_COUNT = 0


def apply_ft_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    copy=False,
    return_orig_weights=False,
    beta_hse=0,
    alpha=0,
    save_weights=True,
    dataset='None',
    batch_size=100,
    DATASET_CONFIG=None,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
    :return: (1) the updated model, (2) the weights that changed
    """
    global GLOBAL_EDIT_COUNT
    GLOBAL_EDIT_COUNT += 1
    cur_edit_count = GLOBAL_EDIT_COUNT

    weights_copy = {}
    modified_weights = {}

    if save_weights:
        save_path = f'./Edited_Weight/FT/{hparams.model_name.split("/")[-1]}/{dataset}_weight_data_batch_{batch_size}_{beta_hse}_{alpha}/'
        os.makedirs(save_path, exist_ok=True)

    if copy:
        model = deepcopy(model)

    deltas = execute_ft(model, tok, requests, hparams)

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, w_name)

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

def project_B_into_soft_dirs(A, B, energy_ratio=0.8, alpha=0.8):
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


def execute_ft(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    requests = deepcopy(requests)
    for request in requests:
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            request["target_new"]["str"] = " " + request["target_new"]["str"]
        print(
            f"Executing FT algo for: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    print(f"Weights to be updated: {list(weights.keys())}")

    # Define inputs
    texts = [r["prompt"].format(r["subject"]) for r in requests]
    targets = [r["target_new"]["str"] for r in requests]

    # Configure optimizer / gradients
    wd = (
        hparams.weight_decay
        if not isinstance(hparams.wd_power_law, tuple)
        else (len(requests) ** hparams.wd_power_law[0])
        * np.exp(hparams.wd_power_law[1])
    )
    print(f"Using weight decay of {wd} for {len(requests)} edits")
    opt = torch.optim.Adam(
        [v for _, v in weights.items()],
        lr=hparams.lr,
        weight_decay=wd,
    )
    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    # Update loop: intervene at layers simultaneously
    loss_meter = AverageMeter()
    for it in range(hparams.num_steps):
        print(20 * "=")
        print(f"Epoch: {it}")
        print(20 * "=")
        loss_meter.reset()

        for txt, tgt in zip(
            chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)
        ):
            inputs = tok(txt, return_tensors="pt", padding=True).to("cuda")
            target_ids = tok(tgt, return_tensors="pt", padding=True)["input_ids"].to(
                "cuda"
            )
            last_token_inds = inputs["attention_mask"].sum(dim=1) - 1
            if tok.unk_token_id is not None:
                loss_mask = torch.ne(target_ids, tok.unk_token_id)
            else:
                loss_mask = torch.ones_like(target_ids, dtype=torch.bool)

            opt.zero_grad()
            bs = inputs["input_ids"].shape[0]
            probs = torch.nn.functional.log_softmax(
                model(**inputs).logits[torch.arange(bs), last_token_inds], dim=-1
            )
            loss = -(torch.gather(probs, 1, target_ids) * loss_mask).sum(
                1
            ) / loss_mask.sum(1)
            loss = loss.mean()
            print(f"Batch loss {loss.item()}")
            loss_meter.update(loss.item(), n=bs)

            if loss.item() >= 1e-2:
                loss.backward()
                opt.step()

            if type(hparams.norm_constraint) is float:
                eps = hparams.norm_constraint
                with torch.no_grad():
                    for k, v in weights.items():
                        v[...] = torch.clamp(
                            v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
                        )

        print(f"Total loss {loss_meter.avg}")

        if loss_meter.avg < 1e-2:
            break

    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
