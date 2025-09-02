from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments

AVAILABLE_OPTIMIZED_PARAMS = ["full", "lora", "only_eos_embedding"]


@dataclass
class SentenceTrainingArguments(TrainingArguments):

    ddp_find_unused_parameters: bool = field(default=False)
    load_best_model_at_end: bool = field(default=False)

    number_of_eos_tokens: int = field(default=1)

    output_dir: str = field(
        default="llama_for_sequential_numbers",
    )
    learning_rate: float = field(default=2e-4)
    max_grad_norm: float = field(default=1.0)

    limit_dataset_shards: int = field(default=0)
    offset_dataset_shards: int = field(default=0)

    warmup_steps: int = field(default=500)
    per_device_train_batch_size: int = field(default=32)
    per_device_eval_batch_size: int = field(default=4)
    num_train_epochs: int = field(default=1)

    lr_scheduler_type: str = field(default="constant_with_warmup")

    average_tokens_across_devices: bool = field(default=True)

    model_checkpoint: str = field(default="")

    weight_decay: float = field(default=0.01)
    eval_strategy: str = field(default="steps")
    eval_steps: int = field(default=10000)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=10000)
    save_total_limit: Optional[int] = field(default=3)
    save_only_model: bool = field(default=True)

    push_to_hub: bool = field(default=False)
    optim: str = field(default="adamw_torch_fused")
    report_to: str = field(default="wandb")  # clearml | wandb | none
    logging_steps: int = field(default=100)
    dataloader_drop_last: bool = field(default=True)
    dataloader_num_workers: int = field(default=0)

    bf16: bool = field(default=False)

    optimized_params: str = field(default="full")  # checkout AVAILABLE_OPTIMIZED_PARAMS

    model_type: str = "dummy"  # dummy | pretrained | SmolLM-1.7B

    hcg_loss_weight: float = 0.0
    hcg_loss_weight_dynamic: bool = False

    hcg_loss_max_value: float = 0.0

    select_train_dataset_items: int = 20000
    add_end_of_sentence_token: bool = field(default=False)

    # When enabled (with multiple EOS tokens), disable loss on EOS tokens
    flexible_eos_tokens: bool = field(default=False)
