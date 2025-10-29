import json
import os
import random
import re
from typing import Dict, Literal

import numpy as np
import torch
from transformers import AutoTokenizer
from wonderwords import RandomWord


def generate_random_sample_full(
    num_examples=200,
    random_word=None,
    hint_first=False,
    niddle_type: Literal["numbers", "strings", "strings_upper"] = "numbers",
    query_words_count=3,
    value_words_count=1,
    value_max_number=1000000,
):

    if random_word is None:
        random_word = RandomWord()

    keys_values = {}
    values_set = set()
    for _ in range(num_examples):
        word = "-".join(random_word.random_words(query_words_count))
        while word in keys_values:
            word = "-".join(random_word.random_words(query_words_count))

        if niddle_type == "numbers":
            value = random.randint(1, value_max_number)
        elif niddle_type == "strings":
            value = "-".join(random_word.random_words(value_words_count))
            while value in values_set:
                value = "-".join(random_word.random_words(value_words_count))
            values_set.add(value)
        elif niddle_type == "strings_upper":
            value = "-".join(random_word.random_words(value_words_count))
            while value in values_set:
                value = "-".join(random_word.random_words(value_words_count))
            value = value.upper()
            values_set.add(value)
        else:
            raise ValueError(f"Invalid niddle type: {niddle_type}")

        keys_values[word] = value

    query_key = random.choice(list(keys_values.keys()))
    query_answer = keys_values[query_key]

    if niddle_type == "numbers":
        magic_object_name = "number"
    elif niddle_type == "strings":
        magic_object_name = "string"
    else:
        raise ValueError(f"Invalid niddle type: {niddle_type}")

    haystack_samples_list = []
    keys_values_keys = list(keys_values.keys())
    random.shuffle(keys_values_keys)
    for key in keys_values_keys:
        value = keys_values[key]

        example = f"One of the special magic {magic_object_name}s for {key} is: {value}.\n"
        haystack_samples_list.append(example)

    haystack_samples = "\n".join(haystack_samples_list)

    template_prefix = f"A special magic {magic_object_name} is hidden within the following text."
    if hint_first:
        template_prefix = f"The special magic {magic_object_name} for {query_key} is hidden within the following text."

    full_sample_template = """
{template_prefix} Make sure to memorize it. I will quiz you about the {magic_object_name} afterwards.
{haystack_samples}
The special magic {magic_object_name} for {query_key} mentioned in the provided text is {template_query_answer}"""

    full_sample_template_no_answer = full_sample_template.format(
        magic_object_name=magic_object_name,
        template_prefix=template_prefix,
        haystack_samples=haystack_samples,
        query_key=query_key,
        template_query_answer="",
    )
    full_sample_template_with_answer = full_sample_template.format(
        magic_object_name=magic_object_name,
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


def generate_random_sample(
    num_examples=200,
    random_word=None,
    no_answer=False,
    return_answer=False,
    hint_first=False,
    query_words_count=3,
    value_words_count=1,
):

    result = generate_random_sample_full(
        num_examples=num_examples,
        random_word=random_word,
        hint_first=hint_first,
        query_words_count=query_words_count,
        value_words_count=value_words_count,
    )

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
    sample_num_examples: int = 10,
    hint_first: bool = False,
    niddle_type: Literal["numbers", "strings", "strings_upper"] = "numbers",
    query_words_count: int = 3,
    value_words_count: int = 1,
    value_max_number: int = 1000000,
    save_results: bool = False,
    checkpoint_path: str = None,
) -> Dict:
    """
    Compute PPL on PG19 for either SentenceLlama or vanilla Llama and return a JSON-serializable dict.
    The returned dict contains overall ppl, per-sample ppls, aggregated prefix metrics, and diagnostics.
    """

    assert dataset_path is None, "dataset_path is not supported for my_recall"

    if save_results:
        out_dir = os.path.join(checkpoint_path, "evaluation")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir,
            f"synthetic_my_recall_num_examples_{sample_num_examples}_type_{niddle_type}_hint_first_{hint_first}_query_words_count_{query_words_count}_value_words_count_{value_words_count}_value_max_number_{value_max_number}.json",
        )
        if os.path.exists(out_path):
            print("Results file already exists, skipping")
            with open(out_path) as f:
                return json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    random_word = RandomWord()

    prediction_scores = []
    for _ in range(max_samples):
        result = generate_random_sample_full(
            num_examples=sample_num_examples,
            random_word=random_word,
            hint_first=hint_first,
            niddle_type=niddle_type,
            query_words_count=query_words_count,
            value_words_count=value_words_count,
            value_max_number=value_max_number,
        )

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
            # print("inputs.input_ids.shape", inputs["input_ids"].shape)
            outputs = model.generate(**inputs, max_new_tokens=20)
            inputs_ids = inputs["input_ids"]

        prediction = tokenizer.decode(outputs[0, inputs_ids.shape[1] :], skip_special_tokens=True)
        if niddle_type == "numbers":
            predicted_numbers = re.findall(r"\D*(\d+)", prediction)
            predicted_number = 0
            if len(predicted_numbers) > 0:
                predicted_number = predicted_numbers[0]
                # print("prediction", prediction)
                # print("predicted_number", predicted_number)
                # print("answer", answer)
                prediction_scores.append(int(predicted_number) == answer)
            else:
                # print("No numbers found in prediction", prediction)
                prediction_scores.append(False)
        elif niddle_type == "strings":
            predicted_string = prediction.strip().split(": ")[0].split(".")[0]
            # print("prediction", prediction)
            # print("predicted_string", predicted_string)
            # print("answer", answer)
            prediction_scores.append((predicted_string == answer) or (answer in prediction))
            # breakpoint()
        else:
            raise ValueError(f"Invalid niddle type: {niddle_type}")

    result = {
        "model_type": model_type,
        "max_samples": int(max_samples),
        "sample_num_examples": int(sample_num_examples),
        "niddle_type": niddle_type,
        "hint_first": hint_first,
        "query_words_count": int(query_words_count),
        "value_words_count": int(value_words_count),
        "value_max_number": int(value_max_number),
        "accuracy": np.mean(prediction_scores),
        "diagnostics": {
            "max_cuda_memory_gb": float(torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else None,
        },
    }

    if save_results:
        out_dir = os.path.join(checkpoint_path, "evaluation")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir,
            f"synthetic_my_recall_num_examples_{sample_num_examples}_type_{niddle_type}_hint_first_{hint_first}_query_words_count_{query_words_count}_value_words_count_{value_words_count}_value_max_number_{value_max_number}.json",
        )
        if not os.path.exists(out_path):
            with open(out_path, "w") as f:
                json.dump(result, f, indent=4)
            print(f"Results saved to {out_path}")
        else:
            raise ValueError(f"Results file {out_path} already exists")

    return result
