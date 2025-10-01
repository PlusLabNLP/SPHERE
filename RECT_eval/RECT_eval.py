import sys
import json
from RECT_eval.ClosedDomainQA_eval import ClosedDomainQA_Eval
from RECT_eval.NLI_eval import NLIEval
from RECT_eval.OpenDomainQA_eval import OpenDomainQA_Eval
from RECT_eval.reasoning_eval import Reasoning_Eval

from util.perplexity import perplexity
from datasets import load_dataset


class RECTEval():
    def __init__(self, model, tokenizer, number_of_tests=None):
        self.model = model

        self.tokenizer = tokenizer

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.reaonsing_eval = Reasoning_Eval(model, tokenizer, number_of_tests=number_of_tests)

        self.nli_eval = NLIEval(model, tokenizer, number_of_tests=number_of_tests)

        self.opendomainQA_eval = OpenDomainQA_Eval(model, tokenizer, number_of_tests=number_of_tests)

        self.closedomainQA_eval = ClosedDomainQA_Eval(model, tokenizer, number_of_tests=number_of_tests)



    def _save_generations(self, record_path, generations, task):
        # store individual generation file
        output_filename = record_path.replace('.json', '_' + task + '_gen.json')
        with open(output_filename, "w") as f:
            json.dump(generations, f, indent=4)

    def evaluate(self, rect_results, record_path, perplexity_flag=False, reaonsing_flag=False, opendomain_flag=False,
                  closedomain_flag=False, nli_flag=False, gen_len=5):
        if perplexity_flag:
            raw_ds = load_dataset(
                "wikitext",
                dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")["wikitext"],
            )
            rect_results['perplexity'] = perplexity(self.model, self.tokenizer, " ".join(raw_ds["train"]['text'][:20]),
                                                    max_input_length=100)

        if reaonsing_flag:
            result_dict, generations = self.reaonsing_eval.evaluate(print_logs=True, gen_len=5)
            rect_results['reaonsing'] = result_dict
            self._save_generations(record_path, generations, 'reaonsing')

        if opendomain_flag:
            result_dict, generations = self.opendomainQA_eval.evaluate(print_logs=True, gen_len=20)
            rect_results['opendomain'] = result_dict
            self._save_generations(record_path, generations, 'opendomain')

        if closedomain_flag:
            result_dict, generations = self.closedomainQA_eval.evaluate(print_logs=True, gen_len=1)
            rect_results['closedomain'] = result_dict
            self._save_generations(record_path, generations, 'closedomain')

        if nli_flag:
            result_dict, generations = self.nli_eval.evaluate(print_logs=True, gen_len=1)
            rect_results['nli'] = result_dict
            self._save_generations(record_path, generations, 'nli')


        return rect_results






