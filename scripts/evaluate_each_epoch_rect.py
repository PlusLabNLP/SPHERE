
import json

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from util import nethook
from util.globals import *

from RECT_eval.RECT_eval import RECTEval


def evaluate_saved_weights_with_rect(model_path, weights_dir, rect_tasks, number_of_tests):
    glue_save_location = Path(weights_dir) / "rect_eval"
    glue_save_location.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.pad_token = tok.eos_token

    rect_eval = RECTEval(model, tok, number_of_tests=number_of_tests)
    not_skip_epochs = {'500', '1000', '2000', '3000'}

    weight_files = sorted(
        [f for f in Path(weights_dir).glob("edit_*.pth")
         if f.stem.split("_")[1] in not_skip_epochs],
        key=lambda x: int(x.stem.split("_")[1])
    )
    print("\n==> Evaluating original (unmodified) model")
    base_result = {'edit_num': -1}
    base_outfile = glue_save_location / "base.json"
    base_result = rect_eval.evaluate(base_result, str(base_outfile), **rect_tasks)

    with open(str(base_outfile).replace(".json", "_rect.json"), "w") as f:
        json.dump(base_result, f, indent=4)

    for weight_file in weight_files:
        print(f"\n==> Evaluating {weight_file.name}")
        checkpoint = torch.load(weight_file, map_location="cpu")
        weights = checkpoint["weight"]

        with torch.no_grad():
            for name, param in weights.items():
                target_param = nethook.get_parameter(model, name)
                target_param.copy_(param.cuda())

        edit_id = int(weight_file.stem.split("_")[1])
        result = {'edit_num': edit_id}
        out_path = glue_save_location / f"{edit_id}_edit.json"
        result = rect_eval.evaluate(result, str(out_path), **rect_tasks)

        with open(str(out_path).replace(".json", "_rect.json"), "w") as f:
            json.dump(result, f, indent=4)

    print("\n✅ All RECT evaluations completed and saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/cache1/chtan/large_models/Llama-3/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--weights_dir", type=str, default=Path(f'/home/jcgu/qyliu/AlphaEdit/Edited_Weight/AlphaEdit/Llama3-8B/mcf_weight_data_batch_100_0.5_0.5'), help="Directory with edit_*.pth weights")
    parser.add_argument("--num_tests", type=int, default=500, help="Number of evaluation examples per RECT task")
    args = parser.parse_args()

    glue_tasks = {
        'reaonsing_flag': True,
        'opendomain_flag': True,
        'closedomain_flag': True,
        'nli_flag': True,
    }

    evaluate_saved_weights_with_rect(
        model_path=args.model_name,
        weights_dir=args.weights_dir,
        rect_tasks=glue_tasks,
        number_of_tests=args.num_tests,
    )
