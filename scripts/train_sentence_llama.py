import os
import shutil
from pathlib import Path

import datasets
import torch
import torch.profiler
import transformers
from accelerate import PartialState
from datasets import load_dataset
from sentence_attention.trainer.arguments import SentenceTrainingArguments
from sentence_attention.trainer.build_model_tokenizer import build_model_tokenizer
from sentence_attention.trainer.trainer import SentenceTrainer
from sentence_attention.trainer.grpo_trainer import GRPOTrainer
from transformers import TrainerCallback
from transformers.loss.loss_utils import ForCausalLMLossFusedLinearCE

from sentence_attention.models.sentence_llama.modeling_sentence_llama import special_token_mask_to_clothest_token_idx_slow


# print("Setting inductor config for flex sentence attention")
# inductor_config.max_autotune = True
# inductor_config.max_autotune_gemm_backends = "ATEN,CUBLAS,TRITON"

if __name__ == "__main__":

    import subprocess

    subprocess.check_output(["nvidia-smi"])

    hf_parser = transformers.HfArgumentParser(SentenceTrainingArguments)
    (training_args,) = hf_parser.parse_args_into_dataclasses()

    compute_metrics = None

    if training_args.gist_placement == "with_logits":
        # assert its llama3.2 with 8 eos tokens
        # TODO load pretokenized dataset

        model, tokenizer = build_model_tokenizer(training_args)

        assert "Llama-3.2".lower() in training_args.model_checkpoint.lower()

        all_ds = []
        for shard_i in range(48):
            ds = datasets.Dataset.load_from_disk(
                f"/workspace-SR004.nfs2/d.tarasov/sentence_attention/artifacts/data/fineweb_edu_tokenized_with_logits_Llama-3.2-1B_max_length_4096_shard_{shard_i}_of_160/"
            )
            all_ds.append(ds)

        concat_ds = datasets.concatenate_datasets(all_ds).remove_columns("token_logprobs")
        special_token_ids = tokenizer.end_of_sentence_token_ids
        num_eos_tokens = training_args.number_of_eos_tokens
        max_sequence_length = training_args.max_sequence_length

        def crutch_collator(examples):

            columns = ["input_ids", "attention_mask", "special_embeddings_mask", "clothest_end_of_sentence_token_idx"]

            stacked_result = {}
            for column in columns:
                stacked_result[column] = torch.stack([torch.tensor(e[column]) for e in examples])

            labels = stacked_result["input_ids"].clone()
            if tokenizer.pad_token_id is not None:
                labels[labels == tokenizer.pad_token_id] = -100

            stacked_result["labels"] = labels

            return stacked_result

        data_collator = crutch_collator

        train_dataset = concat_ds
        eval_dataset = concat_ds.select(range(min(10, len(concat_ds))))

        total_tokens = 0
        gist_tokens = 0
        for item in train_dataset.select(range(100)):
            total_tokens += torch.tensor(item["attention_mask"]).sum().item()
            gist_tokens += torch.tensor(item["special_embeddings_mask"]).sum().item()

        print("total_tokens", total_tokens)
        print("gist_tokens", gist_tokens)
        print("per sample gist placements", gist_tokens / num_eos_tokens / 100)
        print("per sample length", total_tokens / 100)

    else:
        state = PartialState()
        with state.local_main_process_first():
            if training_args.flexible_eos_tokens:
                assert training_args.number_of_eos_tokens > 1, "--flexible_eos_tokens requires --number_of_eos_tokens > 1"

            model, tokenizer = build_model_tokenizer(training_args)

            data_collator = None

            base_output_dir = os.path.basename(training_args.output_dir)

            os.environ["CLEARML_TASK"] = f"{base_output_dir}"

            dataset_shards_limit = training_args.limit_dataset_shards

            if training_args.dataset == "fineweb_edu":
                fineweb_dataset = load_dataset(
                    "HuggingFaceFW/fineweb-edu",
                    "sample-10BT",
                    num_proc=16,
                    split="train",
                    cache_dir="/workspace-SR004.nfs2/d.tarasov/HF_CACHE",
                )
            elif training_args.dataset == "fineweb_edu_100BT":
                fineweb_dataset = load_dataset(
                    "HuggingFaceFW/fineweb-edu",
                    "sample-100BT",
                    num_proc=16,
                    split="train",
                    cache_dir="/workspace-SR004.nfs2/d.tarasov/HF_CACHE",
                )
            elif training_args.dataset == "my_recall":
                from sentence_attention.evaluation.my_recall import generate_random_sample

                my_recall_path = "/workspace-SR004.nfs2/d.tarasov/sentence_attention/artifacts/data/my_recall_10k"
                if not os.path.exists(my_recall_path):
                    dataset_samples = [{"text": generate_random_sample()} for _ in range(10000)]
                    fineweb_dataset = datasets.Dataset.from_list(dataset_samples)
                    fineweb_dataset.save_to_disk(my_recall_path)
                else:
                    fineweb_dataset = datasets.Dataset.load_from_disk(my_recall_path)
            else:
                raise ValueError(f"invalid dataset name: {training_args.dataset}")

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

            print("training_args.select_train_dataset_items", training_args.select_train_dataset_items)
            if training_args.select_train_dataset_items > 0:
                fineweb_dataset = fineweb_dataset.select(range(training_args.select_train_dataset_items))

            # smollm_corpus = smollm_corpus.train_test_split(test_size=100, seed=1)
            train_dataset = fineweb_dataset
            eval_dataset = fineweb_dataset.select(range(min(10, len(fineweb_dataset))))

        # nested_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        special_token_ids = tokenizer.end_of_sentence_token_ids
        num_eos_tokens = training_args.number_of_eos_tokens
        max_sequence_length = training_args.max_sequence_length

        def crutch_collator(examples):
            tokenized_text = tokenizer(
                [e["text"] for e in examples],
                truncation=True,
                max_length=training_args.max_sequence_length,
                padding="max_length",
                return_tensors="pt",
            )

            if len(tokenized_text["attention_mask"].shape) == 3:
                tokenized_text["attention_mask"] = tokenized_text["attention_mask"].squeeze(1)

            input_ids = tokenized_text.input_ids

            special_embeddings_mask = torch.zeros(input_ids.shape, dtype=torch.bool, device=input_ids.device)

            for special_token_id in special_token_ids:
                special_embeddings_mask = special_embeddings_mask | (input_ids == special_token_id)

            clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(special_embeddings_mask)

            labels = input_ids.clone()
            if tokenizer.pad_token_id is not None:
                labels[labels == tokenizer.pad_token_id] = -100
                if training_args.dataset == "my_recall":
                    ams = tokenized_text["attention_mask"].sum(dim=-1).cpu()
                    for b in range(ams.shape[0]):
                        # crutch optimize only last N tokens
                        num_items = ams[b].item() - 10
                        labels[:, :num_items] = -100

            return {
                **tokenized_text,
                "special_embeddings_mask": special_embeddings_mask,
                "clothest_end_of_sentence_token_idx": clothest_end_of_sentence_token_idx,
                "labels": labels,
            }

        data_collator = crutch_collator

    # breakpoint()

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

                embed_tokens_params = model.model.embed_tokens
                if hasattr(embed_tokens_params, "modules_to_save"):
                    # handle lora case
                    embed_tokens_params = embed_tokens_params.modules_to_save

                for p in embed_tokens_params.parameters():
                    current_grad = p.grad
                    p.grad = torch.zeros_like(p.grad)

                    for unfrozen_idx in unfrozen_idxes:
                        p.grad[unfrozen_idx] = current_grad[unfrozen_idx]

                lm_head_params = model.lm_head
                if hasattr(lm_head_params, "modules_to_save"):
                    # handle lora case
                    lm_head_params = lm_head_params.modules_to_save

                for p in lm_head_params.parameters():
                    current_grad = p.grad
                    p.grad = torch.zeros_like(p.grad)

                    for unfrozen_idx in unfrozen_idxes:
                        p.grad[unfrozen_idx] = current_grad[unfrozen_idx]

                return control

        callbacks.append(ZeroOutGradientsForAllExceptEosEmbedding(model))
        print("ZeroOutGradientsForAllExceptEosEmbedding added to callbacks")

    transformers.logging.set_verbosity_info()

    trainer_kwargs = {
        "callbacks": callbacks,
        "processing_class": tokenizer,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
        "compute_loss_func": ForCausalLMLossFusedLinearCE,
    }

    if training_args.trainer_type == "clm":
        trainer_class = SentenceTrainer
    elif training_args.trainer_type == "grpo":
        trainer_class = GRPOTrainer
    else:
        raise ValueError("unknown trainer_class")

    trainer_class.prepare_model_before_training(model, tokenizer)

    print("trainer_class", trainer_class)
    trainer = trainer_class(model, **trainer_kwargs)

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
