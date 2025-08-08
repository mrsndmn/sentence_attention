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
    * eval base checkpoints
    * rerun eval for Llama EOS only Llama-3.2-3B
    * find missing checkpoints
    * why 4 tokens has bad results?
    * simplify experiemnt names
    * paper table formatting, may be different splits
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