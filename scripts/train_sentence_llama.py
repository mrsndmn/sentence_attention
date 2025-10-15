import os
import shutil
from pathlib import Path

import datasets
import torch
import torch.profiler
import transformers
from accelerate import PartialState
from datasets import Dataset, load_dataset
from sentence_attention.artifacts.experiments import WORKDIR_PREFIX
from sentence_attention.trainer.arguments import SentenceTrainingArguments
from sentence_attention.trainer.build_model_tokenizer import build_model_tokenizer
from sentence_attention.trainer.trainer import SentenceTrainer
from transformers import DataCollatorForLanguageModeling, TrainerCallback
from transformers.loss.loss_utils import ForCausalLMLoss

# print("Setting inductor config for flex sentence attention")
# inductor_config.max_autotune = True
# inductor_config.max_autotune_gemm_backends = "ATEN,CUBLAS,TRITON"

if __name__ == "__main__":

    import subprocess

    subprocess.check_output(["nvidia-smi"])

    hf_parser = transformers.HfArgumentParser(SentenceTrainingArguments)
    (training_args,) = hf_parser.parse_args_into_dataclasses()

    state = PartialState()
    with state.local_main_process_first():
        if training_args.flexible_eos_tokens:
            assert training_args.number_of_eos_tokens > 1, "--flexible_eos_tokens requires --number_of_eos_tokens > 1"

        model, tokenizer = build_model_tokenizer(training_args)

        compute_metrics = None
        data_collator = None

        base_output_dir = os.path.basename(training_args.output_dir)

        os.environ["CLEARML_TASK"] = f"{base_output_dir}"

        dataset_shards_limit = training_args.limit_dataset_shards

        if training_args.add_end_of_sentence_token:
            print("Loading fineweb edu tokenized with eos tokenizer")
            datasets_path_prefix = WORKDIR_PREFIX + "/artifacts/data"

            max_length_dataset_suffix = "_max_length_4096"

            dataset_suffix = f"_num_{training_args.number_of_eos_tokens}"

            if training_args.model_type == "sentence_pretrained_checkpoint":
                # dataset_path = f'{current_dir}/fineweb_edu_tokenized_gpt2_with_special_embedding_mask_clothest_eos_token_idx'
                if "llama" in training_args.model_checkpoint.lower():
                    if training_args.dataset == "fineweb_edu":
                        dataset_path = f"{datasets_path_prefix}/fineweb_edu_tokenized_Llama-3.2-1B{max_length_dataset_suffix}_with_eos_token{dataset_suffix}_merged"
                    elif training_args.dataset == "dclm":
                        # dataset_path = f"{datasets_path_prefix}/dclm_tokenized_Llama-3.2-1B_max_length_16384_with_eos_token{dataset_suffix}_merged"
                        dataset_path = f"{datasets_path_prefix}/dclm_tokenized_Llama-3.2-1B_max_length_8192_with_eos_token{dataset_suffix}_merged"
                    elif training_args.dataset == "my_recall":
                        # dataset_path = f"{datasets_path_prefix}/synthetic_niah_tokenized_Llama-3.2-1B_max_length_8192_num_samples_100_with_eos_token{dataset_suffix}_merged"
                        # dataset_path = f"{datasets_path_prefix}/synthetic_niah_tokenized_Llama-3.2-1B_max_length_8192_num_samples_10000_with_eos_token{dataset_suffix}_merged"
                        dataset_path = f"{datasets_path_prefix}/synthetic_niah_tokenized_Llama-3.2-1B_max_length_4096_num_samples_10000_with_eos_token{dataset_suffix}_merged"
                    else:
                        raise ValueError(f"Unknown dataset: {training_args.dataset}")
                elif "qwen2" in training_args.model_checkpoint.lower():
                    dataset_path = f"{datasets_path_prefix}/fineweb_edu_tokenized_Qwen2.5-1.5B{max_length_dataset_suffix}_with_eos_token{dataset_suffix}_merged"
                    print("Increase dataset shards for Qwen2.5-1.5B to", dataset_shards_limit)
                elif "smollm2" in training_args.model_checkpoint.lower():
                    dataset_path = f"{datasets_path_prefix}/fineweb_edu_tokenized_SmolLM2-1.7B{max_length_dataset_suffix}_with_eos_token{dataset_suffix}_merged"
                else:
                    raise ValueError(f"Unknown model checkpoint: {training_args.model_checkpoint}")
                    # dataset_path = f'{current_dir}/fineweb_edu_tokenized_gpt2_with_special_embedding_mask_clothest_eos_token_idx_full'
            else:
                dataset_path = f"{datasets_path_prefix}/fineweb_edu_tokenized_gpt2_eos"

            # dataset_path = f"{datasets_path_prefix}/fineweb_edu_tokenized_Llama-3.2-1B_max_length_8196_with_eos_token_num_4_merged_shard_0_of_1000/"
            # dataset_path = f"{datasets_path_prefix}/fineweb_edu_tokenized_Llama-3.2-1B_max_length_16384_with_eos_token_num_4_merged_shard_0_of_1000/"

            print("Loading dataset from", dataset_path)
            fineweb_dataset = Dataset.load_from_disk(dataset_path)

            TOTAL_SHARDS = 50  # CONSTANT
            dataset_shards = []

            for i in range(TOTAL_SHARDS):
                if i < training_args.offset_dataset_shards or i >= training_args.offset_dataset_shards + dataset_shards_limit:
                    continue
                dataset_shards.append(fineweb_dataset.shard(index=i, num_shards=TOTAL_SHARDS))
                print(f"loading shard {i}")

            print(f"loaded {len(dataset_shards)} shards")
            fineweb_dataset = datasets.concatenate_datasets(dataset_shards)

            print("dataset len", len(fineweb_dataset))

        else:
            data_files = []
            for i in range(6):
                for j in range(10):
                    data_files.append(f"sample/100BT/{i:03}_{j:05}.parquet")

            fineweb_dataset = load_dataset("HuggingFaceFW/fineweb-edu", data_files=data_files, num_proc=48)
            fineweb_dataset = fineweb_dataset["train"]

            def tokenize_function(examples):
                text = examples["text"]

                tokenized_inputs = tokenizer(text, truncation=True, padding="max_length", max_length=1024, return_tensors="pt")

                return tokenized_inputs

            fineweb_dataset = fineweb_dataset.map(tokenize_function, batched=True, num_proc=48)

        print("training_args.select_train_dataset_items", training_args.select_train_dataset_items)
        if training_args.select_train_dataset_items > 0:
            fineweb_dataset = fineweb_dataset.select(range(training_args.select_train_dataset_items))

        # smollm_corpus = smollm_corpus.train_test_split(test_size=100, seed=1)
        train_dataset = fineweb_dataset
        eval_dataset = fineweb_dataset.select(range(min(10, len(fineweb_dataset))))

    nested_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def crutch_collator(examples):
        collate_dummy = nested_data_collator(examples)

        if len(collate_dummy["attention_mask"].shape) == 3:
            collate_dummy["attention_mask"] = collate_dummy["attention_mask"].squeeze(1)

        assert "special_embeddings_mask" in collate_dummy
        assert "clothest_end_of_sentence_token_idx" in collate_dummy

        return collate_dummy

    data_collator = crutch_collator

    trackers_project_name = os.path.basename(training_args.output_dir)
    training_args.run_name = trackers_project_name

    callbacks = []

    class LogModelLayersGradNorm(TrainerCallback):

        def __init__(self, model):
            self.model = model

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            print(
                "model layers up proj grad",
                [
                    (i, self.model.model.layers[i].mlp.up_proj.weight.grad.norm(2).item())
                    for i in range(self.model.config.num_hidden_layers)
                ],
            )
            print(
                "model layers down proj grad",
                [
                    (i, self.model.model.layers[i].mlp.down_proj.weight.grad.norm(2).item())
                    for i in range(self.model.config.num_hidden_layers)
                ],
            )
            print("model lm_head grad norm", self.model.lm_head.weight.grad.norm(2).item())

            print("\n\n\n")
            return control

    # callbacks.append(LogModelLayersGradNorm(model))

    if "only_eos_embedding" in training_args.optimized_params:
        unfrozen_idxes = model.config.end_of_sentence_token_ids

        class ZeroOutGradientsForAllExceptEosEmbedding(TrainerCallback):
            def __init__(self, model):
                self.model = model

            def on_pre_optimizer_step(self, args, state, control, **kwargs):

                for p in model.model.embed_tokens.parameters():
                    current_grad = p.grad
                    p.grad = torch.zeros_like(p.grad)

                    for unfrozen_idx in unfrozen_idxes:
                        p.grad[unfrozen_idx] = current_grad[unfrozen_idx]

                for p in model.lm_head.parameters():
                    current_grad = p.grad
                    p.grad = torch.zeros_like(p.grad)

                    for unfrozen_idx in unfrozen_idxes:
                        p.grad[unfrozen_idx] = current_grad[unfrozen_idx]

                return control

        callbacks.append(ZeroOutGradientsForAllExceptEosEmbedding(model))

    transformers.logging.set_verbosity_info()

    trainer = SentenceTrainer(
        model,
        callbacks=callbacks,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        compute_loss_func=ForCausalLMLoss,
    )

    trainer.accelerator.init_trackers(
        project_name=trackers_project_name,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    out_dir_path = Path(training_args.output_dir)

    if state.is_main_process:
        print("Main process. Moving experiment directory to finished_target_dir")

        finished_target_dir = (
            out_dir_path.parent.parent / "experiments" / f"eos_{training_args.number_of_eos_tokens}" / out_dir_path.name
        )
        print("finished_target_dir", finished_target_dir)
        while True:
            if finished_target_dir.exists():
                finished_target_dir = finished_target_dir.parent / f"{finished_target_dir.name}_duplicate"
                continue
            break

        os.makedirs(finished_target_dir.parent, exist_ok=True)

        shutil.move(out_dir_path, finished_target_dir.parent)
        print("Moved from", out_dir_path, "to finished_target_dir", finished_target_dir)
