# sentence_attention

## TODO

- [ ] datasets - sharding should be done by hf shard, not by hands
- [ ] что делать, чтобы только законченные эксперименты сохранялись? в конце обучения перемещать чекпоинты в другую директорию, а сначала писать в in progress
- [ ] почему лишний токен добавляется в чекпоинт? - scripts/fixscripts/tokenizer_remove_extra_tokens.py
- [ ] support number_of_eos_tokens in training args
- [ ] remove useless arguments from training args
- [ ] Cursor rules
- [ ] validate tests
- [ ] write tests (models, tokenizers, trainer, scripts)
- [ ] github runner action
- [ ] import benchmarks, compute results
- [ ] update environment requirements file


## Submission

- [ ] Submission script - remove metadata, all absolute paths, user nicknames