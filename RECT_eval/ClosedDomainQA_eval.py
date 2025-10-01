import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import jsonlines
import time
import torch
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef
from transformers import AutoTokenizer, AutoModelForCausalLM

class ClosedDomainQA_Eval:
    def __init__(self, model, tokenizer, dataset_path = "./data/task-data/test-ClosedDomainQA.jsonl", number_of_tests=None, number_of_few_shots=0):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.number_of_tests = number_of_tests
        self.number_of_few_shots = number_of_few_shots
        self.postfix_prompt = "answer: "
        self.few_shots = []
        self.eval_dataset = []

        self._load_dataset()
        self._initialize_few_shot_prompt()

    def _load_dataset(self):
        with jsonlines.open(self.dataset_path, "r") as reader:
            all_data = list(reader)
            if self.number_of_tests:
                all_data = all_data[:self.number_of_tests]
            self.eval_dataset = all_data
            self.few_shots = all_data[:self.number_of_few_shots]

    def _initialize_few_shot_prompt(self):
        self.few_shot_context = []
        for few_shot in self.few_shots:
            ans_text = "yes" if few_shot['answer'] else "no"
            context = (
                f"Please answer the given question based on the passage. The answer should be exact 'yes' or 'no'. "
                f"passage: {few_shot['passage']} question: {few_shot['question']}. {self.postfix_prompt} {ans_text}\n\n"
            )
            self.few_shot_context.append(context)

    def _create_prompt(self, example):
        question = (
            f"Please answer the given question based on the passage. The answer should be exact 'yes' or 'no'. "
            f"passage: {example['passage']} question: {example['question']}. {self.postfix_prompt}"
        )
        prompt = ''.join(self.few_shot_context) + question
        label = 1 if example["answer"] else 0
        return prompt, label, example["passage"], example["question"]

    def _get_answer(self, generated_text):
        lower = generated_text.lower()
        if 'yes' in lower:
            return 1
        elif 'no' in lower:
            return 0
        else:
            return -1

    def remove_leading_space_token(self, tok_list, tokenizer):
        """如果第一个 token 是空格，则移除它，否则保留全部"""
        if len(tok_list) > 1 and tokenizer.decode([tok_list[0]]) == " ":
            return tok_list[1:]
        return tok_list


    def evaluate(self, gen_len=1, print_logs=False):
        yes_tok, no_tok = (self.tokenizer(f" {n}")["input_ids"] for n in ['yes', 'no'])

        if "llama" in self.model.config._name_or_path.lower():
            yes_tok = self.remove_leading_space_token(yes_tok, self.tokenizer)
            no_tok = self.remove_leading_space_token(no_tok, self.tokenizer)

        yes_len, no_len = (len(n) for n in [yes_tok, no_tok])
        suffixes = {0: ['yes', yes_tok, yes_len], 1: ['no', no_tok, no_len]}

        correct = 0
        incorrect = 0
        invalid = 0
        pos_correct = 0
        neg_correct = 0
        pos_incorrect = 0
        neg_incorrect = 0

        predictions = []
        labels = []
        predictions_new = []
        stored_generations = []
        start = time.time()

        for s, example in enumerate(self.eval_dataset):
            input_prompt, label, passage, question = self._create_prompt(example)
            print(input_prompt)
            inputs = self.tokenizer(input_prompt, return_tensors='pt', padding=True).to('cuda')
            input_prompt_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            input_prompt_text = self.tokenizer.decode(input_prompt_ids[0], skip_special_tokens=True)

            prefix_tok_len = len(self.tokenizer(input_prompt)["input_ids"])
            if 'llama' in self.model.config._name_or_path.lower():
                prefix_tok_len = prefix_tok_len - 1

            output = self.model.generate(
                input_prompt_ids,
                max_new_tokens=gen_len,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                do_sample=False,
            )

            # # generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            # # answer = self._get_answer(generated_text.replace(input_prompt_text, ''))
            # gen_only_ids = output[0, input_prompt_ids.shape[1]:]
            # generated_text = self.tokenizer.decode(gen_only_ids, skip_special_tokens=True)
            # answer = self._get_answer(generated_text)
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            answer_text = generated_text.split(self.postfix_prompt)[-1].strip()

            answer = self._get_answer(answer_text)

            predictions.append(answer)
            labels.append(label)

            probs = [0 for _ in suffixes.keys()]
            gen_texts = [0 for _ in suffixes.keys()]

            for i in range(len(suffixes.keys())):
                prompt_tok = self.tokenizer([f"{input_prompt} {suffixes[i][0]}"], return_tensors="pt", padding=True).to('cuda')

                with torch.no_grad():
                    logits = self.model(**prompt_tok).logits

                if 'llama' in self.model.config._name_or_path.lower():
                    logits = logits[:, 1:, :]

                cur_len = suffixes[i][2]
                for j in range(cur_len):
                    cur_tok = suffixes[i][1][j]
                    probs[i] += -torch.nn.functional.log_softmax(
                        logits[0, prefix_tok_len + j - 1, :], dim=0
                    )[cur_tok].item()
                probs[i] /= cur_len

                gen_texts[i] = self.tokenizer.decode(
                    logits[0, prefix_tok_len - 1: prefix_tok_len + cur_len - 1, :].argmax(dim=-1)
                )

            prob_yes = np.exp(-probs[0])
            prob_no = np.exp(-probs[1])
            answer_new = 1 if prob_yes > prob_no else 0
            predictions_new.append(answer_new)

            if answer == -1:
                invalid += 1
            else:
                if answer == label:
                    correct += 1
                    if label == 1:
                        pos_correct += 1
                    elif label == 0:
                        neg_correct += 1
                else:
                    incorrect += 1
                    if label == 1:
                        pos_incorrect += 1
                    elif label == 0:
                        neg_incorrect += 1

            exp_temp_dict = {
                'article': passage,
                'question': question,
                'input_prompt': input_prompt_text,
                'true_answer': 'yes' if label == 1 else 'no',
                'generated_text': generated_text.replace(input_prompt_text, ''),
                'answer': answer,
                'correct': answer == label,
                'prob_yes': prob_yes,
                'prob_no': prob_no,
                'highest_probability_answer': 'yes' if answer_new == 1 else 'no',
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

    evaluator = ClosedDomainQA_Eval(
        model=model,
        tokenizer=tokenizer,
        dataset_path="../data/task-data/test-ClosedDomainQA.jsonl",
        number_of_tests=100,
        number_of_few_shots=0
    )

    results, generations = evaluator.evaluate(print_logs=True)
    print(results)


# maxtoken = 1: {'correct': 54, 'incorrect': 9, 'invalid': 37, 'total': 100, 'f1': 0.6435772357723576, 'f1_new': 0.8069930069930069, 'mcc': 0.2835412360886984, 'time': 57.63578915596008}
# maxtoken = 3: {'correct': 62, 'incorrect': 16, 'invalid': 22, 'total': 100, 'f1': 0.6678260869565219, 'f1_new': 0.8069930069930069, 'mcc': 0.2541100264784539, 'time': 62.30215883255005}
# 8 {'correct': 70, 'incorrect': 17, 'invalid': 13, 'total': 100, 'f1': 0.7318309859154931, 'f1_new': 0.8069930069930069, 'mcc': 0.3503829243323882, 'time': 75.36482286453247}
