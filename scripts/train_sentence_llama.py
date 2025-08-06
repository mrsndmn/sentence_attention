from tqdm import tqdm
from dataclasses import dataclass, field
import torch

from transformers import TrainerCallback
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from transformers.trainer import _is_peft_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

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

from transformers import Trainer
from transformers import TrainingArguments

import os
import torch
import torch.nn as nn
from transformers import Trainer
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional, Union, Any, Union, Dict, Tuple

import numpy as np

from dataclasses import dataclass, field
import transformers
from transformers import GenerationConfig
from transformers.trainer import nested_detach
from transformers.trainer_pt_utils import EvalLoopContainer, find_batch_size, IterableDatasetShard
from transformers.trainer_utils import has_length, denumpify_detensorize, EvalLoopOutput, PREFIX_CHECKPOINT_DIR

from torch.utils.data import DataLoader

import time

import torch
from typing import Any, Dict
from typing import List, Optional

import torch.profiler

from transformers.models.llama.extra_types import AVAILABLE_OPTIMIZED_PARAMS

@dataclass
class SentenceTrainingArguments(TrainingArguments):

    ddp_find_unused_parameters: bool = field(default=False)
    load_best_model_at_end: bool = field(default=False)

    output_dir: str = field(default="llama_for_sequential_numbers",)
    learning_rate: float = field(default=2e-4)
    max_grad_norm: float = field(default=1.0)

    dataset: str = field(default='smollm-corpus')
    limit_dataset_shards: int = field(default=0)
    offset_dataset_shards: int = field(default=0)

    warmup_steps: int = field(default=500)
    per_device_train_batch_size: int = field(default=32)
    per_device_eval_batch_size: int = field(default=4)
    num_train_epochs: int = field(default=1)

    lr_scheduler_type: str = field(default='constant_with_warmup')

    average_tokens_across_devices: bool = field(default=True)

    model_checkpoint: str = field(default='')

    weight_decay: float = field(default=0.01)
    eval_strategy: str = field(default="steps")
    eval_steps: int = field(default=10000)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=10000)
    save_total_limit: Optional[int] = field(default=3)
    save_only_model: bool = field(default=True)

    push_to_hub: bool = field(default=False)
    optim: str = field(default="adamw_torch_fused")
    report_to: str = field(default="clearml")
    logging_steps: int = field(default=100)
    dataloader_drop_last: bool = field(default=True)
    dataloader_num_workers: int = field(default=0)

    bf16: bool = field(default=False)

    optimized_params: str = field(default='full') # checkout AVAILABLE_OPTIMIZED_PARAMS

    model_type: str = "dummy" # dummy | pretrained | SmolLM-1.7B

    hcg_loss_weight: float = 0.0
    hcg_loss_weight_dynamic: bool = False

    hcg_loss_max_value: float = 0.0

    select_train_dataset_items: int = 20000
    add_end_of_sentence_token: bool = field(default=False)


class SentenceTrainingArguments(Trainer):


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, log_metrics=True, log_prefix='debug', force_log=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        # if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:

        labels = inputs.pop("labels")

        special_embeddings_mask = inputs.get('special_embeddings_mask')

        attention_mask = inputs['attention_mask']
        token_frequency = inputs.get('token_frequency', None)
        model_kwargs = {
            "input_ids": inputs['input_ids'],
            "attention_mask": attention_mask,
            # "token_frequency": token_frequency,
            "use_cache": False,
            "output_attentions": False,
        }

        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            model_kwargs = {**model_kwargs, **loss_kwargs}

        assert special_embeddings_mask is not None
        model_kwargs["special_embeddings_mask"] = special_embeddings_mask

        model_kwargs['clothest_end_of_sentence_token_idx'] = inputs['clothest_end_of_sentence_token_idx']

        assert special_embeddings_mask.shape == attention_mask.shape

        outputs = model(**model_kwargs)
        # [ bs, seq_len, 2 ]

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None and self.label_smoother is not None or self.compute_loss_func is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()

            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs.logits, labels, vocab_size=unwrapped_model.config.vocab_size, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
        ):
            loss *= self.accelerator.num_processes

        outputs.loss = loss

        return (loss, outputs) if return_outputs else loss

    def update_eval_set_kwargs_containers(self, model, inputs):

        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0
        forced_eos_token_id = eos_token_id

        if self.processing_class is not None:
            bos_token_id = self.processing_class.bos_token_id
            eos_token_id = self.processing_class.eos_token_id
            pad_token_id = self.processing_class.pad_token_id
            forced_eos_token_id = eos_token_id

        gen_params = {
            "do_sample": False,
            "early_stopping": False,
            "num_beams": 1,
            "repetition_penalty": 2.5,
            "remove_invalid_values": True,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "forced_eos_token_id": forced_eos_token_id,
            "use_cache": False,
            "no_repeat_ngram_size": 4,
            "num_return_sequences": 1,
        }
        genconfig = GenerationConfig()

        caption_legth = inputs['input_ids'].shape[1] - 2
        genconfig.max_length = caption_legth

        batch_size, seq_len = inputs['input_ids'].shape[0], 2
        special_embeddings_mask = inputs.get('special_embeddings_mask', None)
        attention_mask = torch.ones([batch_size, seq_len], device=inputs['input_ids'].device)

        prefix_ids = inputs['input_ids'][:, :2]
        all_generation_params = {
            'generation_config': genconfig,
            'max_new_tokens': caption_legth,
            'inputs': prefix_ids,
            'special_embeddings_mask': special_embeddings_mask,
            'attention_mask': attention_mask,
            **gen_params,
        }

        result = {
            "prefix_ids": prefix_ids,
            "input_ids": inputs['input_ids'],
        }

        return result

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """

        # print("inputs", inputs.keys())
        # breakpoint()

        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []


        with torch.no_grad():
            # print('inputs shape', inputs['input_ids'].shape)

            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(
                    model,
                    inputs,
                    return_outputs=True,
                    log_metrics=False,
                    log_prefix='eval_debug',
                    force_log=False,
                )
            loss = loss.mean().detach()

            logits = outputs.logits

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        labels = None

        return (loss, logits, labels)

    @torch.compiler.disable(recursive=True)
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=0)

        metrics = None
        eval_set_kwargs = {}

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = (
                self._prepare_input(inputs[main_input_name]) if "inputs" in args.include_for_metrics else None
            )

            # Update containers
            if losses is not None:
                losses = self.gather_function((losses.repeat(batch_size)))
                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function((inputs_decode))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if labels is not None:
                # Pad labels here, preparing for preprocess_logits_for_metrics in next logits block.
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function((logits))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = self.gather_function((labels))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)

            extra_eval_set_kwargs = self.update_eval_set_kwargs_containers(model, inputs)
            for key, value in extra_eval_set_kwargs.items():
                if key not in eval_set_kwargs:
                    eval_set_kwargs[key] = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=0)

                eval_set_kwargs[key].add(value)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.args.batch_eval_metrics:
                if self.compute_metrics is not None and logits is not None and labels is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    batch_kwargs = {}
                    batch_kwargs["losses"] = losses if "loss" in args.include_for_metrics else None
                    batch_kwargs["inputs"] = inputs if "inputs" in args.include_for_metrics else None
                    metrics = self.compute_metrics(
                        predictions=all_preds,
                        label_ids=all_labels,
                        **eval_set_kwargs
                    )

                del losses, logits, labels, inputs, eval_set_kwargs
                torch.cuda.empty_cache()

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()

                for key in eval_set_kwargs.keys():
                    eval_set_kwargs[key].to_cpu_and_numpy()

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()
        eval_set_kwargs_arrays = dict()
        for key, value in eval_set_kwargs.items():
            eval_set_kwargs_arrays[key] = eval_set_kwargs[key].get_arrays()

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if (
            self.compute_metrics is not None
            # and all_preds is not None
            # and all_labels is not None
            and not self.args.batch_eval_metrics
        ):
            eval_set_kwargs_arrays["losses"] = all_losses if "loss" in args.include_for_metrics else None
            eval_set_kwargs_arrays["inputs"] = all_inputs if "inputs" in args.include_for_metrics else None
            metrics = self.compute_metrics(
                predictions=all_preds,
                label_ids=all_labels,
                **eval_set_kwargs_arrays
            )
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):

        while True:
            try:
                super().save_model(output_dir, _internal_call)
                break
            except Exception as e:
                print("Error in saving model", e)
                time.sleep(300)

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
            elif isinstance(tokenizer, GPT2TokenizerFast) and isinstance(model, LlamaForCausalLM) and 'slm2' in training_args.output_dir:
                print("Loading fineweb edu tokenized with gpt2")
                current_dir = '/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out'

                dataset_path = f'{current_dir}/fineweb_edu_tokenized_gpt2'
                output_dir = sorted(os.listdir(dataset_path))[:30]
                print(output_dir)

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

        if 'special_embeddings_mask' not in collate_dummy:
            collate_dummy['special_embeddings_mask'] = collate_dummy['attention_mask'].cumsum(-1)
            collate_dummy['special_embeddings_mask'][ collate_dummy['special_embeddings_mask'] > 1 ] = 0

            if training_args.add_end_of_sentence_token:
                end_of_sentence_token_id = tokenizer.convert_tokens_to_ids('<end_of_sentence>')
                collate_dummy['special_embeddings_mask'][ collate_dummy['input_ids'] == end_of_sentence_token_id ] = 1

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


    trainer = SentenceTrainingArguments(
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

