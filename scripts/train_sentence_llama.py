from tqdm import tqdm
from dataclasses import dataclass, field
import torch

from transformers import TrainerCallback
from transformers.models.llama.modeling_llama import LlamaForCausalLM


from transformers.loss.loss_utils import ForCausalLMLoss

from transformers.models.llama.modeling_sentence_llama import SentenceLlamaForCausalLM
from transformers.models.llama.tokenization_llama_fast import EOSTokenizerFast
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFastEOS, GPT2TokenizerFast

from transformers.tokenization_utils_fast import PreTrainedTokenizerFastEOS
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFastEOS
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFastEOS

from transformers.models.qwen2.modeling_sentence_qwen2 import SentenceQwen2ForCausalLM

from datasets import load_dataset, Dataset
import datasets
from accelerate import PartialState

from transformers import GenerationConfig

import torch
import transformers
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

import os
import torch
import torch.nn as nn
import torch

import torch.profiler

from sentence_attention.trainer.arguments import AVAILABLE_OPTIMIZED_PARAMS

from sentence_attention.trainer.arguments import SentenceTrainingArguments
from sentence_attention.trainer.trainer import SentenceTrainer


def freeze_model(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def build_model(training_args: SentenceTrainingArguments):
    tokenizer = None
    model_checkpoint = training_args.model_checkpoint

    if training_args.add_end_of_sentence_token:

        tokenizer_class = type(AutoTokenizer.from_pretrained(model_checkpoint)).__name__

        if tokenizer_class in ['GPT2TokenizerFast', 'GPT2TokenizerFastEOS']:
            tokenizer_class = GPT2TokenizerFastEOS
        elif tokenizer_class in ['PreTrainedTokenizerFast', 'PreTrainedTokenizerFastEOS']:
            tokenizer_class = PreTrainedTokenizerFastEOS
        elif tokenizer_class in ['Qwen2TokenizerFast', 'Qwen2TokenizerFastEOS']:
            tokenizer_class = Qwen2TokenizerFastEOS
        else:
            raise ValueError(f"Invalid tokenizer class: {tokenizer_class}")

        print("tokenizer_class", tokenizer_class)
        tokenizer = tokenizer_class.from_pretrained(model_checkpoint)

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    print("tokenizer", tokenizer)

    torch_dtype = torch.bfloat16

    if training_args.model_type == 'sentence_pretrained_checkpoint':
        model_checkpoint = training_args.model_checkpoint
        print("Load sentence llama model from", model_checkpoint)
        model_class = None
        if 'lama' in model_checkpoint.lower() or 'smollm2' in model_checkpoint.lower():
            model_class = SentenceLlamaForCausalLM
        elif 'qwen' in model_checkpoint.lower():
            model_class = SentenceQwen2ForCausalLM

        print("model_class", model_class)
        model = model_class.from_pretrained(model_checkpoint, torch_dtype=torch_dtype)

        model.config._attn_implementation = 'sentence_attention'
    else:
        raise ValueError(f"{training_args.model_type} is not supported")

    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    if training_args.add_end_of_sentence_token and model.config.vocab_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resized model embeddings to vocabulary size: {len(tokenizer)}")
        model.config.end_of_sentence_token_id = tokenizer.convert_tokens_to_ids('<end_of_sentence>')
        print("model.config.end_of_sentence_token_id", model.config.end_of_sentence_token_id)

    if training_args.model_type == "sentence_pretrained_checkpoint":
        optimized_params = training_args.optimized_params
        print("optimized_params", optimized_params)

        assert optimized_params in AVAILABLE_OPTIMIZED_PARAMS, f'unknown optimized_params value: {optimized_params}. available ones: {AVAILABLE_OPTIMIZED_PARAMS}'

        if 'full' == optimized_params:
            assert len(optimized_params) == 1
        elif 'only_eos_embedding' == optimized_params:
            freeze_model(model)
            for p in model.model.embed_tokens.parameters():
                p.requires_grad = True

            for p in model.lm_head.parameters():
                p.requires_grad = True
        elif 'lora' == optimized_params:
            from peft import LoraConfig, TaskType, get_peft_model

            # create LoRA configuration object
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, # type of task to train on
                inference_mode=False, # set to False for training
                r=8, # dimension of the smaller matrices
                lora_alpha=32, # scaling factor
                lora_dropout=0.1 # dropout of LoRA layers
            )
            model.add_adapter(lora_config, adapter_name="lora_1")

        else:
            raise ValueError()

    # print("force full fp32 training!")
    # model = model.to(torch.float32)

    print("model", type(model))
    print("num trainable model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("num freezed model parameters:", sum(p.numel() for p in model.parameters() if not p.requires_grad))

    return model, tokenizer

if __name__ == "__main__":

    import subprocess
    subprocess.check_output(['nvidia-smi'])

    hf_parser = transformers.HfArgumentParser(SentenceTrainingArguments)
    (training_args,) = hf_parser.parse_args_into_dataclasses()

    model, tokenizer = build_model(training_args)

    compute_metrics = None
    data_collator = None

    base_output_dir = os.path.basename(training_args.output_dir)

    os.environ['CLEARML_TASK'] = f"{base_output_dir}"

    state = PartialState()
    with state.local_main_process_first():

        if training_args.dataset == 'smollm-corpus':

            if training_args.add_end_of_sentence_token:
                print("Loading fineweb edu tokenized with gpt2_eos")
                current_dir = '/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out'

                if training_args.model_type == 'sentence_pretrained_checkpoint':
                    # dataset_path = f'{current_dir}/fineweb_edu_tokenized_gpt2_with_special_embedding_mask_clothest_eos_token_idx'
                    if 'llama-3.2' in training_args.model_checkpoint.lower():
                        dataset_path = f'{current_dir}/fineweb_edu_tokenized_Llama-3.2-1B_with_eos_token'
                    elif 'qwen2' in training_args.model_checkpoint.lower():
                        dataset_path = f'{current_dir}/fineweb_edu_tokenized_Qwen2.5-1.5B_with_eos_token'
                    elif 'smollm2' in training_args.model_checkpoint.lower():
                        dataset_path = f'{current_dir}/fineweb_edu_tokenized_SmolLM2-1.7B_with_eos_token'
                    else:
                        raise ValueError(f"Unknown model checkpoint: {training_args.model_checkpoint}")
                        # dataset_path = f'{current_dir}/fineweb_edu_tokenized_gpt2_with_special_embedding_mask_clothest_eos_token_idx_full'
                else:
                    dataset_path = f'{current_dir}/fineweb_edu_tokenized_gpt2_eos'

                output_dir = sorted(os.listdir(dataset_path))
                if training_args.limit_dataset_shards > 0:
                    offset = training_args.offset_dataset_shards
                    end_idx = offset + training_args.limit_dataset_shards
                    print("Dataset offset end idx:", offset, ":", end_idx)
                    output_dir = output_dir[offset:end_idx]

                print("loading dataset", dataset_path, 'with', len(output_dir), 'dataset shards', output_dir)

                all_datasets = []
                for data_file in tqdm(output_dir, desc='Loading datasets'):
                    dataset = Dataset.load_from_disk(f'{dataset_path}/{data_file}')
                    all_datasets.append(dataset)

                smollm_corpus = datasets.concatenate_datasets(all_datasets)
            else:
                data_files = []
                for i in range(6):
                    for j in range(10):
                        data_files.append(f"sample/100BT/{i:03}_{j:05}.parquet")

                smollm_corpus = load_dataset("HuggingFaceFW/fineweb-edu", data_files=data_files, num_proc=48)
                smollm_corpus = smollm_corpus['train']


                def tokenize_function(examples):
                    text = examples['text']

                    tokenized_inputs = tokenizer(text, truncation=True, padding='max_length', max_length=1024, return_tensors='pt')

                    return tokenized_inputs

                smollm_corpus = smollm_corpus.map(tokenize_function, batched=True, num_proc=48)

            print("training_args.select_train_dataset_items", training_args.select_train_dataset_items)
            if training_args.select_train_dataset_items > 0:
                smollm_corpus = smollm_corpus.select(range(training_args.select_train_dataset_items))

            # smollm_corpus = smollm_corpus.train_test_split(test_size=100, seed=1)
            train_dataset = smollm_corpus
            eval_dataset = smollm_corpus.select(range(100))

    nested_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


    def crutch_collator(examples):
        collate_dummy = nested_data_collator(examples)

        if len(collate_dummy['attention_mask'].shape) == 3:
            collate_dummy['attention_mask'] = collate_dummy['attention_mask'].squeeze(1)

        assert 'special_embeddings_mask' in collate_dummy
        assert 'clothest_end_of_sentence_token_idx' in collate_dummy

        return collate_dummy

    data_collator = crutch_collator

    trackers_project_name = os.path.basename(training_args.output_dir)
    training_args.run_name = trackers_project_name

    callbacks = []


    class LogModelLayersGradNorm(TrainerCallback):

        def __init__(self, model):
            self.model = model

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            print("model layers up proj grad", [ (i, self.model.model.layers[i].mlp.up_proj.weight.grad.norm(2).item()) for i in range(self.model.config.num_hidden_layers) ])
            print("model layers down proj grad", [ (i, self.model.model.layers[i].mlp.down_proj.weight.grad.norm(2).item()) for i in range(self.model.config.num_hidden_layers) ])
            print("model lm_head grad norm", self.model.lm_head.weight.grad.norm(2).item())

            print("\n\n\n")
            return control

    # callbacks.append(LogModelLayersGradNorm(model))

    if 'only_eos_embedding' in training_args.optimized_params:
        unfrozen_idx = model.config.end_of_sentence_token_id

        class ZeroOutGradientsForAllExceptEosEmbedding(TrainerCallback):
            def __init__(self, model):
                self.model = model

            def on_pre_optimizer_step(self, args, state, control, **kwargs):

                for p in model.model.embed_tokens.parameters():
                    current_grad = p.grad
                    p.grad = torch.zeros_like(p.grad)
                    p.grad[unfrozen_idx] = current_grad[unfrozen_idx]

                for p in model.lm_head.parameters():
                    current_grad = p.grad
                    p.grad = torch.zeros_like(p.grad)
                    p.grad[unfrozen_idx] = current_grad[unfrozen_idx]

                return control

        callbacks.append(ZeroOutGradientsForAllExceptEosEmbedding(model))


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

    trainer.train()

