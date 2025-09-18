import argparse
import os

import transformers
from peft import PeftConfig, PeftModel
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM, Qwen2ForCausalLM

from sentence_attention.evaluation.benchmarks import all_benchmarks
from sentence_attention.evaluation.evaluation import evaluate_lighteval_task, evaluate_lighteval_task_save_results
from sentence_attention.evaluation.pg19 import evaluate_pg19_ppl, save_pg19_results_json
from sentence_attention.models.sentence_llama.modeling_sentence_llama import SentenceLlamaForCausalLM
from sentence_attention.models.sentence_qwen2.modeling_sentence_qwen2 import SentenceQwen2ForCausalLM


def load_model_from_checkpoint(checkpoint_path):

    config = AutoConfig.from_pretrained(checkpoint_path)
    model_class_name = config.architectures[0]

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    if hasattr(tokenizer, "num_eos_tokens"):
        print(
            "tokenizer num_eos_tokens",
            tokenizer.num_eos_tokens,
            "end_of_sentence_token_ids",
            tokenizer.end_of_sentence_token_ids,
        )
    else:
        print("tokenizer does not have num_eos_tokens", type(tokenizer))

    if model_class_name == "SentenceLlamaForCausalLM":
        model_class = SentenceLlamaForCausalLM
    elif model_class_name == "SentenceQwen2ForCausalLM":
        model_class = SentenceQwen2ForCausalLM
    elif model_class_name == "LlamaForCausalLM":
        model_class = LlamaForCausalLM
    elif model_class_name == "Qwen2ForCausalLM":
        model_class = Qwen2ForCausalLM
    else:
        raise ValueError(f"Model class {model_class_name} not supported")

    model = model_class.from_pretrained(checkpoint_path)
    model.eval()

    return model, tokenizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sel-llm", action="store_true", default=False)
    parser.add_argument("--benchmark", type=str, required=True, choices=all_benchmarks)
    parser.add_argument("--no-save-results", action="store_true", default=False)

    args = parser.parse_args()

    if "lora" in args.checkpoint:
        peft_config = PeftConfig.from_pretrained(args.checkpoint)
        base_model, tokenizer = load_model_from_checkpoint(peft_config.base_model_name_or_path)
        model = PeftModel.from_pretrained(base_model, args.checkpoint)
        model = model.merge_and_unload()
    else:
        model, tokenizer = load_model_from_checkpoint(args.checkpoint)

    if args.sel_llm:
        assert transformers.__version__ == "4.53.0", "transformers version must be 4.53.0"
        import os

        os.environ["SEPCACHE_ENABLED"] = "1"

    if args.benchmark == "pg19":

        model_type = "vanilla"
        if "sentence" in str(type(model)).lower():
            model_type = "sentence"

        results = evaluate_pg19_ppl(
            model,
            tokenizer,
            dataset_path="/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/pg19_test",  # TODO move to HF?
            model_type=model_type,
            max_samples=-1,
            max_length=32000,
            # max_length=64000,
        )
        if not args.no_save_results:
            out_dir = os.path.join(args.checkpoint, "evaluation")
            os.makedirs(out_dir, exist_ok=True)
            out_path = save_pg19_results_json(out_dir, results)
            print(f"Results saved to {out_path}")
    else:
        if args.no_save_results:
            print("Evaluating without saving results")
            results = evaluate_lighteval_task(model, args.benchmark)
        else:
            results = evaluate_lighteval_task_save_results(model, args.checkpoint, args.benchmark)

    print(results)
