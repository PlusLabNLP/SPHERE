import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# ok
import time
import torch
import numpy as np
import jsonlines
from sklearn.metrics import f1_score, matthews_corrcoef
from transformers import AutoTokenizer, AutoModelForCausalLM


class Reasoning_Eval:
    def __init__(self, model, tokenizer, dataset_path = "./data/task-data/test-reasoning.jsonl", number_of_tests=None):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.number_of_tests = number_of_tests
        self.eval_dataset = []

        self._load_dataset()

    def _load_dataset(self):
        with jsonlines.open(self.dataset_path, "r") as reader:
            all_data = list(reader)
        if self.number_of_tests:
            all_data = all_data[:self.number_of_tests]
        self.eval_dataset = all_data

    def _create_prompt(self, example):
        """
        构造 Reasoning 任务的 prompt
        """
        question = example["question"]
        answer_full = example["answer"]
        hint = answer_full.split("#### ")[0]
        gold_answer = answer_full.split("#### ")[1]
        prompt = f"Q: {question} A:Let's think step by step. {hint} Therefore, the answer (arabic numerals) is:"
        return prompt, gold_answer

    def evaluate(self, gen_len=5, print_logs=False):
        correct = incorrect = 0
        predictions, labels = [], []
        stored_generations = []
        start = time.time()

        for idx, example in enumerate(self.eval_dataset):
            prompt, gold_answer = self._create_prompt(example)

            inputs = self.tokenizer(prompt, return_tensors='pt', padding=True).to('cuda')

            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=gen_len,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                do_sample=False
            )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted_answer = generated_text.split("Therefore, the answer (arabic numerals) is:")[-1].strip()

            is_correct = gold_answer in predicted_answer
            if is_correct:
                correct += 1
            else:
                incorrect += 1

            predictions.append(predicted_answer)
            labels.append(gold_answer)

            stored_generations.append({
                'question': example['question'],
                'gold_answer': gold_answer,
                'prompt': prompt,
                'generated_text': generated_text.replace(prompt, ''),
                'predicted_answer': predicted_answer,
                'correct': is_correct
            })

            if print_logs:
                print(f"[{idx+1}] GT: {gold_answer} | Pred: {predicted_answer} | Correct: {is_correct}")

        end = time.time()
        total = correct + incorrect
        acc = correct / total if total > 0 else 0.0
        mcc = matthews_corrcoef(
            [1 if labels[i] in predictions[i] else 0 for i in range(len(labels))],
            [1] * len(labels)
        )
        f1 = f1_score(
            [1 if labels[i] in predictions[i] else 0 for i in range(len(labels))],
            [1] * len(labels),
            average='weighted'
        )

        result_dict = {
            'correct': correct,
            'incorrect': incorrect,
            'total': len(self.eval_dataset),
            'acc': acc,
            'f1': f1,
            'mcc': mcc,
            'time': end - start,
        }
        return result_dict, stored_generations


# if __name__ == '__main__':
#     model_path = "/cache1/chtan/large_models/Llama-3/Meta-Llama-3-8B-Instruct"
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     tokenizer.pad_token_id = tokenizer.eos_token_id
#     tokenizer.padding_side = 'left'
#     model = AutoModelForCausalLM.from_pretrained(model_path).to('cuda')
#
#     evaluator = Reasoning_Eval(
#         model=model,
#         tokenizer=tokenizer,
#         dataset_path="../data/task-data/test-reasoning.jsonl",
#         number_of_tests=100  # 可选
#     )
#
#     results, generations = evaluator.evaluate(print_logs=True)
#     print(results)
