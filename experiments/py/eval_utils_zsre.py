"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import AttributeSnippets

def remove_leading_space_token(tok_list, tokenizer):
    """如果第一个 token 是空格，则移除它，否则保留全部"""
    if len(tok_list) > 1 and tokenizer.decode([tok_list[0]]) == " ":
        return tok_list[1:]
    return tok_list

def compute_rewrite_quality_zsre(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    snips: AttributeSnippets,
    vec: TfidfVectorizer,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    paraphrase_prompts = record["paraphrase_prompts"]
    neighborhood_prompts = record["neighborhood_prompts"]

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
    ]
    # Flatten all the evaluated prefixes into one list.
    target_tok = tok(" " + target_new["str"])["input_ids"]
    # if 'llama' in model.config._name_or_path.lower():
    #     target_tok = target_tok[1:]
    if 'llama' in model.config._name_or_path.lower():
        target_tok = remove_leading_space_token(target_tok, tok)

    inp_prompts_og = list(chain(*prob_prompts))

    inp_prompts = [
        el + tok.decode(target_tok[:i]) if 'llama' not in model.config._name_or_path.lower() or i ==0 else el + ' ' + tok.decode(target_tok[:i])
        for el in inp_prompts_og
        for i in range(len(target_tok))
    ]
    inp_targets = [
        tok.decode(target_tok[i])
        for _ in range(len(inp_prompts_og))
        for i in range(len(target_tok))
    ]

    stuff_probs = test_batch_prediction_acc(model, tok, inp_prompts, inp_targets)

    # Predict for neighborhood prompts (dictionary format).
    neighborhood_correct = test_batch_prediction_acc(
        model,
        tok,
        [
            el["prompt"].format(record["requested_rewrite"])
            for el in neighborhood_prompts
        ],
        [el["target"] for el in neighborhood_prompts],
    )

    probs = stuff_probs + neighborhood_correct

    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(
        [l * len(target_tok) for l in map(len, prob_prompts)]
    ).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    # Structure the restuls as a dictionary.
    ret = {
        f"{key}_correct": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
            ]
        )
    }
    ret["neighborhood_prompts_correct"] = neighborhood_correct

    return ret


def test_batch_prediction_acc(model, tok, prompts: typing.List[str], target):
    # 安全性检查：防止空 prompt
    if not prompts or len(prompts) == 0 or all(p.strip() == "" for p in prompts):
        print("[ERROR] Empty or blank prompts passed to tokenizer.")
        return []

    # 编码 prompts
    try:
        prompt_tok = tok(prompts, padding=True, return_tensors="pt").to("cuda")
    except Exception as e:
        print(f"[ERROR] Tokenization failed for prompts: {e}")
        return []

    prompt_tok = tok(
        prompts,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        logits = model(**prompt_tok).logits
        last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
        to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
        gathered = torch.gather(logits, 1, to_gather).squeeze(1)
        ans = torch.argmax(gathered, dim=1)

        correct_id = tok(target, padding=True, return_tensors="pt").to("cuda")[
            "input_ids"
        ]
        # # Temporary hack to deal with foreign characters.
        # if 'llama' in model.config._name_or_path.lower():
        #     correct_id = correct_id[:, 1].squeeze()
        # else:
        #     correct_id = correct_id[:, 0].squeeze()
        # LLaMA 模型可能带有 prefix 空格 token，谨慎移除
        if 'llama' in model.config._name_or_path.lower():
            if correct_id.shape[1] > 1 and tok.decode([correct_id[0, 0]]) == " ":
                correct_id = correct_id[:, 1]
            else:
                correct_id = correct_id[:, 0]
        else:
            correct_id = correct_id[:, 0]

        return (ans == correct_id).detach().cpu().numpy().tolist()
