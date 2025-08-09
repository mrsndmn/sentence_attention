# sentence_attention

## Installation

```bash
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest -v ./tests

# Run black
black .

# Run ruff
ruff check .

# Run flake8
flake8 .
```


## Tasks

- [ ] eval all checkpoints, plot graphs
    * paper table formatting, may be different splits
    * run new experiments with 4 eos tokens for Qwen2.5-1.5B, Llama3.2-3B, etc, 8B models
    * how metrics change during training stages?
- [ ] Fix auto modeling. create separate config classes for llama and qwen2
- [ ] remove useless arguments from training args
- [ ] refactor run jobs script - add arguments to scripts, move common code to separate file
- [ ] Cursor rules
- [ ] write more tests? (models, tokenizers, trainer, scripts)
- [ ] github runner action
- [ ] import benchmarks, compute results
- [ ] update environment requirements file


## Submission

- [ ] Submission script - remove metadata, all absolute paths, user nicknames