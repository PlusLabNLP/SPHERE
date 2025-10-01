import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import regex
import string
import time
import torch
import numpy as np
import jsonlines
from sklearn.metrics import f1_score, matthews_corrcoef
from transformers import AutoTokenizer, AutoModelForCausalLM


def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])


class OpenDomainQA_Eval():
    def __init__(self, model, tokenizer, dataset_path = "./data/task-data/test-OpenDomainQA.jsonl", number_of_tests=None, number_of_few_shots=0):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.number_of_tests = number_of_tests
        self.number_of_few_shots = number_of_few_shots
        self.postfix_prompt = "Answer:"
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
            answer = normalize_answer(few_shot['answer'][0])  # only use the first
            context = (
                f"Refer to the passage below and answer the following question.\n"
                f"Passage: {few_shot['output'][0]}\n"
                f"Question: {few_shot['question']}\n"
                f"{self.postfix_prompt} {answer}\n\n"
            )
            self.few_shot_context.append(context)

    def _create_prompt(self, example):
        context = (
            f"Refer to the passage below and answer the following question.\n"
            f"Passage: {example['output'][0]}\n"
            f"Question: {example['question']}\n"
            f"{self.postfix_prompt}"
        )
        prompt = ''.join(self.few_shot_context) + context
        return prompt, example["question"], example["output"][0], example["answer"]

    def _soft_exact_match(self, normalized_output, ground_truths):
        if normalized_output is None:
            return False
        text = str(normalized_output).strip()
        if not text:
            return False

        words = normalized_output.split()
        if ems(words[0], ground_truths) or ems(words[-1], ground_truths):
            return True
        for i in range(len(words)):
            output = words[i]
            if ems(output, ground_truths):
                return True
            for j in range(i + 1, len(words)):
                output = output + " " + words[j]
                if ems(output, ground_truths):
                    return True
        return False

    def remove_leading_space_token(self, tok_list, tokenizer):
        if len(tok_list) > 1 and tokenizer.decode([tok_list[0]]) == " ":
            return tok_list[1:]
        return tok_list

    def evaluate(self, gen_len=20, print_logs=False):
        correct = 0
        incorrect = 0
        invalid = 0
        stored_generations = []
        start = time.time()
        output_lengths = []

        for s, example in enumerate(self.eval_dataset):
            prompt, question, passage, ground_truth_answers = self._create_prompt(example)
            inputs = self.tokenizer(prompt, return_tensors='pt').to('cuda')
            input_prompt_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            # max_len = input_prompt_ids.shape[1] + gen_len
            outputs = self.model.generate(
                input_ids=input_prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_len,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                do_sample=False,
            )

            # generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # input_prompt_text = self.tokenizer.decode(input_prompt_ids[0], skip_special_tokens=True)
            # generated_answer = generated_text.replace(input_prompt_text, '').replace('\n', ' ').strip()
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_answer = generated_text.split(self.postfix_prompt)[-1].strip()

            normalized_output = normalize_answer(generated_answer)
            normalized_gt = [normalize_answer(a) for a in ground_truth_answers]

            is_correct = self._soft_exact_match(normalized_output, normalized_gt)

            if is_correct:
                correct += 1
            else:
                incorrect += 1

            output_lengths.append(len(normalized_output.split()))

            stored_generations.append({
                'passage': passage,
                'question': question,
                'true_answer': ground_truth_answers,
                'input_prompt': prompt,
                'generated_text': generated_answer,
                'normalized_output': normalized_output,
                'correct': is_correct,
            })

            if print_logs:
                print(f"[{s+1}] EM: {is_correct} | Predict: {normalized_output} | GT: {normalized_gt}")
                print('--' * 30)

        end = time.time()
        total = len(self.eval_dataset)
        avg_len = np.mean(output_lengths) if output_lengths else 0.0
        total = correct + incorrect

        result_dict = {
            'correct': correct,
            'incorrect': incorrect,
            'invalid': 0,
            'total': total,
            'acc': correct / total if total > 0 else 0.0,
            'f1': 0.0,
            'f1_new': 0.0,
            'mcc': 0.0,
            'time': end - start,
            'avg_answer_length': avg_len,
        }

        return result_dict, stored_generations


if __name__ == '__main__':
    model_path = "/cache1/chtan/large_models/Llama-3/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(model_path).to('cuda')

    evaluator = OpenDomainQA_Eval(
        model=model,
        tokenizer=tokenizer,
        dataset_path="../data/task-data/test-OpenDomainQA.jsonl",
        number_of_tests=100,
        number_of_few_shots=0
    )

    results, generations = evaluator.evaluate(print_logs=True)
    print(results)
