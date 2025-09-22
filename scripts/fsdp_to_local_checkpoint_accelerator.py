import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate import Accelerator
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint import load as tdc_load_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from transformers import AutoModelForCausalLM, AutoTokenizer

from sentence_attention.tokenization_utils_fast import PreTrainedTokenizerFastEOS


def find_fsdp_shard_dir(checkpoint_dir: Path, fsdp_dir_arg: str | None) -> Path:
    if fsdp_dir_arg:
        shard_dir = Path(fsdp_dir_arg)
        if shard_dir.is_dir():
            return shard_dir
        # allow relative to checkpoint_dir
        shard_dir = checkpoint_dir / fsdp_dir_arg
        if shard_dir.is_dir():
            return shard_dir
        raise FileNotFoundError(f"FSDP shards directory not found: {fsdp_dir_arg}")

    # autodetect a subdir like pytorch_model_fsdp_0 containing __*.distcp
    candidates = [p for p in checkpoint_dir.iterdir() if p.is_dir() and re.match(r"^pytorch_model_fsdp_\d+$", p.name)]
    for cand in sorted(candidates):
        if any(f.suffix == ".distcp" for f in cand.iterdir() if f.is_file()):
            return cand
    # also allow direct pointing to a directory of distcp files
    if any(f.suffix == ".distcp" for f in checkpoint_dir.iterdir() if f.is_file()):
        return checkpoint_dir
    raise FileNotFoundError(
        f"Could not locate a directory with .distcp shards under {checkpoint_dir}. "
        f"Expected a subdirectory like pytorch_model_fsdp_0/"
    )


def init_distributed_if_needed(backend: str = "gloo") -> None:
    if dist.is_available() and not dist.is_initialized():
        # initialize a local single-process process group
        tmp_dir = tempfile.mkdtemp(prefix="fsdp_local_pg_")
        init_file = Path(tmp_dir) / "shared_init"
        # ensure path exists
        init_method = f"file://{init_file}"
        dist.init_process_group(backend=backend, init_method=init_method, rank=0, world_size=1)


def build_model(
    config_source: str | Path, base_model: str | Path, dtype: torch.dtype | None, tokenizer
) -> AutoModelForCausalLM:
    torch_dtype = dtype

    from sentence_attention.models.sentence_llama.modeling_sentence_llama import SentenceLlamaForCausalLM

    # Build model from the checkpoint's config to preserve tie_word_embeddings and vocab_size
    model = SentenceLlamaForCausalLM.from_pretrained(str(base_model), torch_dtype=torch_dtype)

    # Ensure embeddings match tokenizer length if provided
    if tokenizer is not None and hasattr(tokenizer, "__len__"):
        tok_len = len(tokenizer)
        if getattr(model.config, "vocab_size", None) != tok_len:
            model.resize_token_embeddings(tok_len)
            model.config.vocab_size = tok_len

    return model


def try_load_tokenizer(tokenizer_source: Path | str | None):
    if tokenizer_source is None:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(str(tokenizer_source), use_fast=True)
        return tok
    except Exception:
        return None


def tdc_available() -> bool:
    return FileSystemReader is not None and tdc_load_state_dict is not None


def load_fsdp_distcp_into_model(fsdp_shard_dir: Path, model: AutoModelForCausalLM, device: torch.device) -> None:
    if not tdc_available():
        raise RuntimeError(
            "torch.distributed.checkpoint is not available. Please install PyTorch >= 2.0 with distributed checkpoint support."
        )
    if FSDP is None or StateDictType is None:
        raise RuntimeError("torch.distributed.fsdp is required to load FSDP sharded checkpoints.")

    print(f"Loading FSDP distcp shards from: {fsdp_shard_dir}")

    # Initialize a single-process process group so FSDP APIs are usable
    init_distributed_if_needed(backend="gloo")

    # Ensure model on the expected compute device for FSDP
    model.to(device)

    # Wrap with FSDP to materialize expected FSDP state_dict structure
    # Note: with world_size=1 this does not actually shard parameters
    fsdp_kwargs = {"use_orig_params": True}
    if device.type == "cuda":
        fsdp_kwargs["device_id"] = torch.cuda.current_device()
    fsdp_model = FSDP(model, **fsdp_kwargs)
    # Expect SHARDED_STATE_DICT format which matches distcp layout created by FSDP integration
    FSDP.set_state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT)

    def _attempt_load(fsdp_wrapped: nn.Module, attempt_name: str) -> bool:
        FSDP.set_state_dict_type(fsdp_wrapped, StateDictType.SHARDED_STATE_DICT)
        target_state = FSDP.state_dict(fsdp_wrapped)
        reader = FileSystemReader(str(fsdp_shard_dir))
        # tdc_load(state_dict=target_state, storage_reader=reader)
        # reader.

        tdc_load_state_dict(state_dict=target_state, storage_reader=reader)
        FSDP.load_state_dict(fsdp_wrapped, target_state)
        return True

    # 1) Try with an extra 'model._orig_mod.' prefix
    container1 = nn.Module()
    container1.model = nn.Module()
    container1.model._orig_mod = model
    container1.to(device)
    fsdp_container1 = FSDP(container1, **fsdp_kwargs)
    _attempt_load(fsdp_container1, attempt_name="container.model._orig_mod prefix")

    model.load_state_dict(container1.model._orig_mod.state_dict(), strict=False)
    return

    # Unwrap not strictly necessary here since we copied into original model


def parse_dtype(arg: str | None) -> torch.dtype | None:
    if arg is None:
        return None
    arg = arg.lower()
    if arg in {"float16", "fp16", "half"}:
        return torch.float16
    if arg in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if arg in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {arg}")


def copy_if_exists(src_dir: Path, dst_dir: Path, names: list[str]) -> None:
    for name in names:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert FSDP distcp checkpoint to a standard Hugging Face checkpoint")
    parser.add_argument(
        "--checkpoint-dir", type=str, required=True, help="Path to HF checkpoint dir containing config.json and fsdp shards"
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Where to save the consolidated HF checkpoint")
    parser.add_argument(
        "--fsdp-dir", type=str, default=None, help="Path to directory with .distcp shards (defaults to autodetect)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="HF model id or local dir to take config/tokenizer from if not in checkpoint",
    )
    parser.add_argument("--dtype", type=str, default=None, help="Optional dtype: float16|bfloat16|float32")
    parser.add_argument(
        "--save-tokenizer-from", type=str, default=None, help="Optional path/model id to load tokenizer from and save alongside"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for FSDP load: auto|cuda|cpu. Default auto prefers CUDA if available",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # dcp_to_torch_save not works
    # from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
    # dcp_to_torch_save(checkpoint_dir, output_dir)
    # return

    fsdp_shard_dir = find_fsdp_shard_dir(checkpoint_dir, args.fsdp_dir)

    # Determine config/tokenizer source
    config_source: Path | str | None = None
    tokenizer_source: Path | str | None = None
    if (checkpoint_dir / "config.json").is_file():
        config_source = checkpoint_dir
    elif args.base_model is not None:
        config_source = args.base_model
    else:
        raise FileNotFoundError("config.json not found in checkpoint dir and --base-model not provided")

    if args.save_tokenizer_from is not None:
        tokenizer_source = args.save_tokenizer_from
    elif (checkpoint_dir / "tokenizer.json").is_file() or (checkpoint_dir / "tokenizer.model").is_file():
        tokenizer_source = checkpoint_dir
    elif args.base_model is not None:
        tokenizer_source = args.base_model

    dtype = parse_dtype(args.dtype)

    print(f"Using config from: {config_source}")
    if tokenizer_source is not None:
        print(f"Using tokenizer from: {tokenizer_source}")
    print(f"FSDP shards directory: {fsdp_shard_dir}")

    num_eos_tokens = int(re.search(r"num_eos_tokens_(\d+)", args.checkpoint_dir).group(1))

    assert "llama" in args.checkpoint_dir.lower(), "only llama checkpoints are supported"
    # Prefer tokenizer saved with the checkpoint/base model; fall back to EOS-augmented tokenizer only if needed
    tokenizer = try_load_tokenizer(tokenizer_source)
    if tokenizer is None:
        tokenizer = PreTrainedTokenizerFastEOS.from_pretrained(args.base_model, num_eos_tokens=num_eos_tokens)

    # Build the model from the checkpoint's config to match training settings (e.g., tie_word_embeddings)
    model = build_model(config_source, args.base_model, dtype, tokenizer)
    # Ensure tied embeddings before loading if the checkpoint was saved with tying (avoids zero-sized lm_head in distcp)
    if hasattr(model, "tie_weights"):
        # Prefer config flag if already set; otherwise force-enable to match typical LLaMA training
        if getattr(model.config, "tie_word_embeddings", True) is not True:
            model.config.tie_word_embeddings = True
        model.tie_weights()
    model.eval()

    from accelerate import FullyShardedDataParallelPlugin
    from accelerate.utils.fsdp_utils import load_fsdp_model
    from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType

    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_type=StateDictType.SHARDED_STATE_DICT, state_dict_config=ShardedStateDictConfig(offload_to_cpu=False)
    )

    accelerator = Accelerator()
    container1 = nn.Module()
    container1._orig_mod = model
    container1 = accelerator.prepare(container1)

    load_fsdp_model(
        fsdp_plugin,
        accelerator,
        container1,
        args.checkpoint_dir,
    )

    # accelerator.load_state( args.checkpoint_dir )

    breakpoint()


if __name__ == "__main__":
    # Honor environment hints if present

    os.environ.setdefault("HF_HOME", "/workspace-SR004.nfs2/.cache/huggingface")
    # Allow running from repo root where src is present
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    main()
