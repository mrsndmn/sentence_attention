import json
import os
import random
import re
from typing import Dict, Literal

import numpy as np
import torch
from transformers import AutoTokenizer
from wonderwords import RandomWord


def generate_random_sample_full(num_examples=200, random_word=None, hint_first=False):

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

    haystack_samples = "\n".join(haystack_samples_list)

    template_prefix = "A special magic number is hidden within the following text."
    if hint_first:
        template_prefix = f"The special magic number for {key} is hidden within the following text."

    full_sample_template = """
{template_prefix} Make sure to memorize it. I will quiz you about the number afterwards.
{haystack_samples}
The special magic number for {query_key} mentioned in the provided text is {template_query_answer}"""

    full_sample_template_no_answer = full_sample_template.format(
        template_prefix=template_prefix, haystack_samples=haystack_samples, query_key=query_key, template_query_answer=""
    )
    full_sample_template_with_answer = full_sample_template.format(
        template_prefix=template_prefix,
        haystack_samples=haystack_samples,
        query_key=query_key,
        template_query_answer=f"{query_answer}.\n",
    )

    return {
        "sample_with_answer": full_sample_template_with_answer,
        "sample_without_answer": full_sample_template_no_answer,
        "query_answer": query_answer,
    }


def generate_random_sample(num_examples=200, random_word=None, no_answer=False, return_answer=False, hint_first=False):

    result = generate_random_sample_full(num_examples=num_examples, random_word=random_word, hint_first=hint_first)

    if not return_answer:
        if no_answer:
            return result["sample_without_answer"]
        return result["sample_with_answer"]

    if no_answer:
        return result["sample_without_answer"], result["query_answer"]

    return result["sample_with_answer"], result["query_answer"]


@torch.no_grad()
def evaluate_synthetic_my_recall(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    dataset_path=None,
    model_type: Literal["sentence", "vanilla"] = "sentence",
    max_samples: int = 10,
    hint_first: bool = False,
) -> Dict:
    """
    Compute PPL on PG19 for either SentenceLlama or vanilla Llama and return a JSON-serializable dict.
    The returned dict contains overall ppl, per-sample ppls, aggregated prefix metrics, and diagnostics.
    """

    assert dataset_path is None, "dataset_path is not supported for my_recall"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    random_word = RandomWord()

    prediction_scores = []
    for _ in range(max_samples):
        result = generate_random_sample_full(num_examples=10, random_word=random_word, hint_first=hint_first)

        sample = result["sample_without_answer"]
        answer = result["query_answer"]

        if "LlamaAutoCompressorModel" in str(type(model)):
            sample_sentences = sample.split(".")
            context = ".".join(sample_sentences[:-1])
            prompt_suffix = sample_sentences[-1]

            inputs = tokenizer(context, return_tensors="pt").to(device)
            context_tokens = tokenizer(context, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
            prompt_tokens = tokenizer(prompt_suffix, add_special_tokens=False, return_tensors="pt").input_ids.cuda()

            summary_vectors = model(context_tokens, output_softprompt=True).softprompt
            print("summary_vectors", summary_vectors.shape)
            outputs = model.generate(prompt_tokens, do_sample=False, softprompt=summary_vectors, max_new_tokens=20)
            inputs_ids = prompt_tokens
        else:
            inputs = tokenizer(sample, return_tensors="pt").to(device)
            print("inputs.input_ids.shape", inputs["input_ids"].shape)
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
