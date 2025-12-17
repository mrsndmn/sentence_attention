import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm


from dataclasses import asdict


from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import nltk

from sentence_attention.boundaries.predictor import BolmoConfig, BolmoLocalEncoder


def boundary_logprobs_to_logits(boundary_logprobs: torch.Tensor) -> torch.Tensor:
    probs = torch.exp(torch.clamp(boundary_logprobs, max=0))
    probs = probs.clamp(min=1e-6, max=1 - 1e-6)
    logits = torch.log(probs) - torch.log1p(-probs)
    return logits


def batch_precision_recall_fscore(
    pred_mask: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[float, float, float]:
    valid_tokens = attention_mask.bool()
    preds = pred_mask & valid_tokens
    gold = labels.bool() & valid_tokens

    true_pos = (preds & gold).sum().item()
    predicted_pos = preds.sum().item()
    gold_pos = gold.sum().item()

    precision = true_pos / (predicted_pos + 1e-8)
    recall = true_pos / (gold_pos + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def train_boundary_predictor(
    model: BolmoLocalEncoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_interval: int,
    balance_boundary_loss: bool,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    writer: SummaryWriter | None = None,
    global_step_offset: int = 0,
):
    model.train()
    running_loss = 0.0
    running_f1 = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_batches = 0

    pbar = tqdm(dataloader, desc="train", leave=False)
    for global_step, batch in enumerate(pbar):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        boundary_labels = batch["boundary_labels"].to(device)

        boundary_logprobs, boundary_mask = model(input_ids)
        boundary_logits = boundary_logprobs_to_logits(boundary_logprobs)

        pos_weight = None
        if balance_boundary_loss:
            pos_count = (boundary_labels * attention_mask).sum()
            neg_count = ((1 - boundary_labels) * attention_mask).sum()
            pos_weight = (neg_count / (pos_count + 1e-8)).clamp(min=1.0)

        loss = F.binary_cross_entropy_with_logits(
            boundary_logits,
            boundary_labels.float(),
            pos_weight=pos_weight,
            reduction="none",
        )
        loss = (loss * attention_mask).sum() / attention_mask.sum().clamp(min=1)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            precision, recall, f1 = batch_precision_recall_fscore(boundary_mask, boundary_labels, attention_mask)

        running_loss += loss.item()
        running_f1 += f1
        running_precision += precision
        running_recall += recall
        running_batches += 1

        if global_step % log_interval == 0:
            avg_loss = running_loss / running_batches
            avg_f1 = running_f1 / running_batches
            avg_precision = running_precision / running_batches
            avg_recall = running_recall / running_batches
            lr_val = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                f1=f"{avg_f1:.4f}",
                prec=f"{avg_precision:.4f}",
                rec=f"{avg_recall:.4f}",
                lr=f"{lr_val:.2e}",
            )
            if writer is not None:
                step = global_step_offset + global_step
                writer.add_scalar("train/loss", avg_loss, step)
                writer.add_scalar("train/f1", avg_f1, step)
                writer.add_scalar("train/precision", avg_precision, step)
                writer.add_scalar("train/recall", avg_recall, step)
                writer.add_scalar("train/lr", lr_val, step)
            running_loss = 0.0
            running_f1 = 0.0
            running_precision = 0.0
            running_recall = 0.0
            running_batches = 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pretrain boundaries model")
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help="Path or name of the model checkpoint to use",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=4096,
        help="Maximum sequence length for tokenization",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="HuggingFaceFW/fineweb-edu",
        help="Dataset name to load",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="sample-10BT",
        help="Dataset configuration name",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/workspace-SR004.nfs2/d.tarasov/HF_CACHE",
        help="Cache directory for datasets",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=16,
        help="Number of processes for dataset processing",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Optimizer learning rate",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Steps between logging loss and f-score",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=0,
        help="Limit training samples for a quick run (0 for all)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for the training DataLoader",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save the trained boundary predictor",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile on the model for speed",
    )
    args = parser.parse_args()

    model_checkpoint = args.model_checkpoint
    checkpoint_name = model_checkpoint.replace("/", "__")

    output_dir = args.output_dir or os.path.join("artifacts", "experiments", "bp", checkpoint_name)
    os.makedirs(output_dir, exist_ok=True)

    # TODO use vocabulary from model_checkpoint in Bolmo model

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    fineweb_dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        num_proc=args.num_proc,
        split="train",
        cache_dir=args.cache_dir,
    )

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    sentence_tokenizer = nltk.tokenize.PunktTokenizer()

    # Load dataloader.

    SENTENCE_ENDINGS = {".", "!", "?"}

    def _last_non_space_index_in_range(text: str, start_idx: int, end_idx: int) -> int:
        idx = end_idx
        while idx >= start_idx and text[idx].isspace():
            idx -= 1
        return idx

    def data_collator(examples):
        texts = [e["text"] for e in examples]

        all_input_ids = []
        all_attention = []
        all_boundary = []

        for text in texts:
            processed_chars: list[str] = []
            boundary_char_positions: list[int] = []
            drop_punct_indices: set[int] = set()

            for sent_start, sent_end in sentence_tokenizer.span_tokenize(text):
                last_idx = _last_non_space_index_in_range(text, sent_start, sent_end - 1)
                if last_idx < sent_start:
                    continue

                drop_punct = text[last_idx] in SENTENCE_ENDINGS and random.random() < 0.5
                boundary_idx = last_idx

                if drop_punct:
                    prev_idx = _last_non_space_index_in_range(text, sent_start, last_idx - 1)
                    if prev_idx < sent_start:
                        drop_punct = False
                    else:
                        boundary_idx = prev_idx
                        drop_punct_indices.add(last_idx)

                boundary_char_positions.append(boundary_idx)

            orig_to_new: dict[int, int] = {}
            for orig_idx, ch in enumerate(text):
                if orig_idx in drop_punct_indices:
                    continue
                orig_to_new[orig_idx] = len(processed_chars)
                processed_chars.append(ch)

            boundary_char_positions_processed = []
            for orig_idx in boundary_char_positions:
                mapped = orig_to_new.get(orig_idx)
                if mapped is not None:
                    boundary_char_positions_processed.append(mapped)

            processed_text = "".join(processed_chars)

            tokenized = tokenizer(
                processed_text,
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_sequence_length,
                return_offsets_mapping=True,
            )

            input_ids = tokenized["input_ids"]
            offsets = tokenized["offset_mapping"]

            boundaries = [0] * len(input_ids)
            for boundary_char_idx in boundary_char_positions_processed:
                for token_idx, (start, end) in enumerate(offsets):
                    if start <= boundary_char_idx < end:
                        boundaries[token_idx] = 1
                        break

            seq_len = min(len(input_ids), args.max_sequence_length)
            pad_len = args.max_sequence_length - seq_len

            input_ids = input_ids[: args.max_sequence_length]
            boundaries = boundaries[: args.max_sequence_length]

            attention = [1] * seq_len + [0] * pad_len

            input_ids.extend([tokenizer.pad_token_id] * pad_len)
            boundaries.extend([0] * pad_len)

            all_input_ids.append(input_ids)
            all_boundary.append(boundaries)
            all_attention.append(attention)

        batch = {
            "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(all_attention, dtype=torch.long),
            "boundary_labels": torch.tensor(all_boundary, dtype=torch.long),
        }
        return batch

    def preview_collator_samples(dataset, num_samples: int = 3):
        if len(dataset) == 0:
            return

        RED, RESET = "\033[91m", "\033[0m"
        sample_indices = list(range(min(num_samples, len(dataset))))
        sample_batch = data_collator([dataset[i] for i in sample_indices])

        print("\nPreview of collator output (boundaries in red):")
        for sample_idx in range(sample_batch["input_ids"].shape[0]):
            input_ids = sample_batch["input_ids"][sample_idx].tolist()
            attention = sample_batch["attention_mask"][sample_idx].tolist()
            boundaries = sample_batch["boundary_labels"][sample_idx].tolist()
            tokens = tokenizer.convert_ids_to_tokens(input_ids)

            pretty_tokens = []
            for tok, attn, boundary in zip(tokens, attention, boundaries):
                if not attn:
                    break
                display_tok = tok.replace("Ġ", " ")
                if boundary:
                    pretty_tokens.append(f"{RED}{display_tok}{RESET}")
                else:
                    pretty_tokens.append(display_tok)

            print(f"sample {sample_idx}: {' '.join(pretty_tokens)}")

    # Initialize simple boundary predictor model and  write training cycle
    # For sentence boundaries prediction.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    config = BolmoConfig()

    with torch.no_grad():
        base_model = AutoModel.from_pretrained(model_checkpoint)
        base_embedding_weight = base_model.get_input_embeddings().weight.detach().clone()
        del base_model
        subword_embedding = nn.Embedding.from_pretrained(base_embedding_weight, freeze=True)

    config.hidden_size = subword_embedding.weight.shape[1]

    model = BolmoLocalEncoder(config, subword_embedding).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tqdm.write(f"Trainable parameters: {trainable_params:,}")

    if args.compile:
        model = torch.compile(model)

    train_dataset = fineweb_dataset
    if args.max_train_samples and args.max_train_samples > 0:
        max_samples = min(args.max_train_samples, len(train_dataset))
        train_dataset = train_dataset.select(range(max_samples))
        print(f"Using {max_samples} samples for training")

    preview_collator_samples(train_dataset)

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    steps_per_epoch = max(len(dataloader), 1)
    total_steps = max(1, steps_per_epoch * args.num_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=5e-5,
    )

    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb"))
    print("writer", writer)

    epoch_iter = tqdm(range(args.num_epochs), desc="epochs")
    for epoch_idx in epoch_iter:
        train_boundary_predictor(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            log_interval=args.log_interval,
            balance_boundary_loss=config.balance_boundary_loss,
            scheduler=scheduler,
            writer=writer,
            global_step_offset=epoch_idx * steps_per_epoch,
        )
        writer.add_scalar("train/epoch", epoch_idx + 1, (epoch_idx + 1) * steps_per_epoch)

    save_path = os.path.join(output_dir, "model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_checkpoint": args.model_checkpoint,
            "config": asdict(config),
            "cli_args": dict(vars(args)),
        },
        save_path,
    )
    tqdm.write(f"Saved boundary model to {save_path}")
    writer.close()
