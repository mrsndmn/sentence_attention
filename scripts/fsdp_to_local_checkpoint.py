import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist

try:
    # PyTorch 2.x distributed checkpoint APIs
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint import load_state_dict as tdc_load_state_dict
except Exception:  # pragma: no cover
    FileSystemReader = None
    tdc_load_state_dict = None

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType
except Exception:  # pragma: no cover
    FSDP = None
    StateDictType = None

from transformers import AutoModelForCausalLM, AutoTokenizer

from sentence_attention.tokenization_utils_fast import PreTrainedTokenizerFastEOS


def _print(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


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


def build_model(base_model: str, dtype: torch.dtype | None, tokenizer) -> AutoModelForCausalLM:
    torch_dtype = dtype

    from sentence_attention.models.sentence_llama.modeling_sentence_llama import SentenceLlamaForCausalLM

    # parse regexp num_eos_tokens_4

    model = SentenceLlamaForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype)
    model.resize_token_embeddings(len(tokenizer))

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

    _print(f"Loading FSDP distcp shards from: {fsdp_shard_dir}")

    # Initialize a single-process process group so FSDP APIs are usable
    init_distributed_if_needed(backend="gloo")

    # Ensure model on the expected compute device for FSDP
    model.to(device)

    # Wrap with FSDP to materialize expected FSDP state_dict structure
    # Note: with world_size=1 this does not actually shard parameters
    fsdp_kwargs = {}
    if device.type == "cuda":
        fsdp_kwargs["device_id"] = torch.cuda.current_device()
    fsdp_model = FSDP(model, **fsdp_kwargs)
    # Expect SHARDED_STATE_DICT format which matches distcp layout created by FSDP integration
    FSDP.set_state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT)

    # Prepare target state_dict skeleton
    target_state_dict = FSDP.state_dict(fsdp_model)

    # Load tensors from distcp into the skeleton
    reader = FileSystemReader(str(fsdp_shard_dir))
    try:
        # PyTorch >= 2.1
        tdc_load_state_dict(state_dict=target_state_dict, storage_reader=reader, no_dist=True)
    except TypeError:
        # PyTorch 2.0 signature
        tdc_load_state_dict(target_state_dict, storage_reader=reader)

    # Apply to model
    FSDP.load_state_dict(fsdp_model, target_state_dict)

    # Unwrap back to plain module
    unwrapped = fsdp_model.module
    # Copy weights into the original model reference to keep caller semantics
    model.load_state_dict(unwrapped.state_dict(), strict=True)


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

    _print(f"Using config from: {config_source}")
    if tokenizer_source is not None:
        _print(f"Using tokenizer from: {tokenizer_source}")
    _print(f"FSDP shards directory: {fsdp_shard_dir}")

    num_eos_tokens = int(re.search(r"num_eos_tokens_(\d+)", args.checkpoint_dir).group(1))

    assert "llama" in args.checkpoint_dir.lower(), "only llama checkpoints are supported"
    tokenizer = PreTrainedTokenizerFastEOS.from_pretrained(args.base_model, num_eos_tokens=num_eos_tokens)

    model = build_model(args.base_model, dtype, tokenizer)
    model.eval()
    # resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested via --device=cuda but not available")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError("--device must be one of: auto|cuda|cpu")

    with torch.no_grad():
        load_fsdp_distcp_into_model(fsdp_shard_dir, model, device)

    # Save model
    _print(f"Saving consolidated model to: {output_dir}")
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    # Also copy generation config if present
    copy_if_exists(checkpoint_dir, output_dir, ["generation_config.json"])

    _print("Done.")


if __name__ == "__main__":
    # Honor environment hints if present
    os.environ.setdefault("HF_HOME", "/workspace-SR004.nfs2/.cache/huggingface")
    # Allow running from repo root where src is present
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    main()
