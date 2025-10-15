import json
import os
import random
import re
from typing import Dict, Literal

import numpy as np
import torch
from transformers import AutoTokenizer
from wonderwords import RandomWord


def generate_random_sample(num_examples=200, random_word=None, no_answer=False, return_answer=True):

    if random_word is None:
        random_word = RandomWord()

    keys_values = {}
    for _ in range(num_examples):
        word = "-".join(random_word.random_words(3))
        while word in keys_values:
            word = "-".join(random_word.random_words(3))
        keys_values[word] = random.randint(1, 1000000)

    query_key = random.choice(list(keys_values.keys()))
    query_answer = keys_values[query_key]

    haystack_samples_list = []
    keys_values_keys = list(keys_values.keys())
    random.shuffle(keys_values_keys)
    for key in keys_values_keys:
        value = keys_values[key]

        example = f"One of the special magic numbers for {key} is: {value}.\n"
        haystack_samples_list.append(example)

    template_query_answer = ""
    if not no_answer:
        template_query_answer = f"{query_answer}.\n"
    haystack_samples = "\n".join(haystack_samples_list)
    full_sample = f"""
A special magic number is hidden within the following text. Make sure to memorize it. I will quiz you about the number afterwards.
{haystack_samples}
The special magic number for {query_key} mentioned in the provided text is {template_query_answer}"""

    if return_answer:
        return full_sample, query_answer

    return full_sample


def evaluate_synthetic_my_recall(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    dataset_path=None,
    model_type: Literal["sentence", "vanilla"] = "sentence",
    max_samples: int = 10,
) -> Dict:
    """
    Compute PPL on PG19 for either SentenceLlama or vanilla Llama and return a JSON-serializable dict.
    The returned dict contains overall ppl, per-sample ppls, aggregated prefix metrics, and diagnostics.
    """

    assert dataset_path is None, "dataset_path is not supported for my_recall"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    prediction_scores = []
    for _ in range(max_samples):
        sample, answer = generate_random_sample(num_examples=200, return_answer=True, no_answer=True)

        inputs = tokenizer(sample, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=20)
        inputs_ids = inputs["input_ids"]
        prediction = tokenizer.decode(outputs[0, inputs_ids.shape[1] :], skip_special_tokens=True)
        predicted_number = re.findall(r"\D*(\d+)", prediction)[0]

        print("prediction", prediction)
        print("predicted_number", predicted_number)
        print("answer", answer)
        prediction_scores.append(int(predicted_number) == answer)

    result = {
        "model_type": model_type,
        "max_samples": int(max_samples),
        "accuracy": np.mean(prediction_scores),
        "diagnostics": {
            "max_cuda_memory_gb": float(torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else None,
        },
    }

    return result


def save_pg19_results_json(output_dir: str, results: Dict) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "pg19.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return out_path
