import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import csv
import time
import torch
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef
from transformers import AutoTokenizer, AutoModelForCausalLM

class NLIEval():
    def __init__(self, model, tokenizer, dataset_path = './data/task-data/test-NLI.tsv', number_of_tests=None, number_of_few_shots=0):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.number_of_tests = number_of_tests
        self.number_of_few_shots = number_of_few_shots
        self.postfix_prompt = "answer:"
        self.few_shots = []
        self.eval_dataset = []

        self._load_dataset()
        self._initialize_few_shot_prompt()

    def _load_dataset(self):
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)  # skip header if any
            all_data = [{"index": line[0], "sentence1": line[1], "sentence2": line[2], "label": line[3]} for line in reader]
            if self.number_of_tests:
                all_data = all_data[:self.number_of_tests]
            self.eval_dataset = all_data
            self.few_shots = all_data[:self.number_of_few_shots]

    def _initialize_few_shot_prompt(self):
        self.few_shot_context = []
        for example in self.few_shots:
            ans = "true" if example["label"] != "not_entailment" else "false"
            context = f"{example['sentence1']} entails {example['sentence2']}. True or False? {self.postfix_prompt} {ans}\n\n"
            self.few_shot_context.append(context)

    def _create_prompt(self, example, gen_len):
        question = f"{example['sentence1']} entails {example['sentence2']}. True or False? {self.postfix_prompt}"
        prompt = ''.join(self.few_shot_context) + question
        label = 1 if example["label"] != "not_entailment" else 0
        return prompt, label, example["sentence1"], example["sentence2"]

    def _get_answer(self, generated_text):
        lower = generated_text.lower()
        if 'true' in lower:
            return 1
        elif 'false' in lower:
            return 0
        return -1

    def remove_leading_space_token(self, tok_list):
        if len(tok_list) > 1 and self.tokenizer.decode([tok_list[0]]) == " ":
            return tok_list[1:]
        return tok_list

    def evaluate(self, gen_len=1, print_logs=False):
        true_tok, false_tok = (self.tokenizer(f" {n}")["input_ids"] for n in ['true', 'false'])

        if "llama" in self.model.config._name_or_path.lower():
            true_tok = self.remove_leading_space_token(true_tok)
            false_tok = self.remove_leading_space_token(false_tok)

        true_len, false_len = (len(n) for n in [true_tok, false_tok])
        suffixes = {1: ['true', true_tok, true_len], 0: ['false', false_tok, false_len]}

        correct = incorrect = invalid = 0
        pos_correct = neg_correct = pos_incorrect = neg_incorrect = 0

        predictions = []
        labels = []
        predictions_new = []
        stored_generations = []

        start = time.time()
        for s, example in enumerate(self.eval_dataset):
            input_prompt, label, sent1, sent2 = self._create_prompt(example, gen_len)
            inputs = self.tokenizer(input_prompt, return_tensors='pt', padding=True).to('cuda')
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            input_prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

            prefix_tok_len = len(self.tokenizer(input_prompt)["input_ids"])
            if 'llama' in self.model.config._name_or_path.lower():
                prefix_tok_len -= 1

            # max_len = input_ids.shape[1] + gen_len
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_len,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                do_sample=False
            )
            # generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            # answer = self._get_answer(generated_text.replace(input_prompt_text, ''))

            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            answer_text = generated_text.split(self.postfix_prompt)[-1].strip()

            answer = self._get_answer(answer_text)


            predictions.append(answer)
            labels.append(label)

            probs = [0 for _ in suffixes]
            for i in suffixes:
                prompt_tok = self.tokenizer([f"{input_prompt} {suffixes[i][0]}"], return_tensors="pt", padding=True).to('cuda')
                with torch.no_grad():
                    logits = self.model(**prompt_tok).logits
                if 'llama' in self.model.config._name_or_path.lower():
                    logits = logits[:, 1:, :]
                cur_len = suffixes[i][2]
                for j in range(cur_len):
                    cur_tok = suffixes[i][1][j]
                    probs[i] += -torch.nn.functional.log_softmax(logits[0, prefix_tok_len + j - 1], dim=0)[cur_tok].item()
                probs[i] /= cur_len

            prob_true = np.exp(-probs[1])
            prob_false = np.exp(-probs[0])
            answer_new = 1 if prob_true > prob_false else 0
            predictions_new.append(answer_new)

            if answer == -1:
                invalid += 1
            else:
                if answer == label:
                    correct += 1
                    if label == 1: pos_correct += 1
                    else: neg_correct += 1
                else:
                    incorrect += 1
                    if label == 1: pos_incorrect += 1
                    else: neg_incorrect += 1

            exp_temp_dict = {
                'sentence1': sent1,
                'sentence2': sent2,
                'input_prompt': input_prompt_text,
                'true_answer': 'true' if label == 1 else 'false',
                'generated_text': generated_text.replace(input_prompt_text, ''),
                'answer': 'true' if answer == 1 else ('false' if answer == 0 else 'invalid'),
                'correct': answer == label,
                'prob_true': prob_true,
                'prob_false': prob_false,
                'highest_probability_answer': 'true' if answer_new == 1 else 'false',
                'correct_new': answer_new == label,
            }
            stored_generations.append(exp_temp_dict)

            if print_logs:
                mcc = matthews_corrcoef(labels, predictions)
                f1 = f1_score(labels, predictions, average='weighted')
                print(generated_text)
                print(correct, incorrect, invalid, s+1, '|', pos_correct, neg_correct,
                      '|', pos_incorrect, neg_incorrect,
                      '|ACC: ', correct / (correct + incorrect + invalid),
                      '|MCC:', mcc, '|F1:', f1)
                print('--' * 50)

        end = time.time()
        mcc = matthews_corrcoef(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        f1_new = f1_score(labels, predictions_new, average='weighted')
        total = correct + incorrect
        result_dict = {
            'correct': correct,
            'incorrect': incorrect,
            'invalid': invalid,
            'total': s+1,
            'acc': correct / total if total > 0 else 0.0,
            'f1': f1,
            'f1_new': f1_new,
            'mcc': mcc,
            'time': end - start,
        }

        return result_dict, stored_generations


if __name__ == '__main__':
    model_path = "/cache1/chtan/large_models/Llama-3/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(model_path).to('cuda')

    evaluator = NLIEval(
        model=model,
        tokenizer=tokenizer,
        dataset_path="../data/task-data/test-NLI.tsv",
        number_of_tests=100,
        number_of_few_shots=0
    )

    results, generations = evaluator.evaluate(print_logs=True)
    print(results)
