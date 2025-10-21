from sentence_attention.models.checkpoint import load_model_from_checkpoint

if __name__ == "__main__":

    base_model = "./artifacts/experiments/eos_4/sentence_Llama-3.2-3B_ft_4k_full_num_eos_tokens_4_KS38WK9A/checkpoint-9067"
    donor_model = "./artifacts/experiments/eos_8/sentence_Llama-3.2-3B_ft_4k_full_num_eos_tokens_8_XF4BDKFN/checkpoint-8000"

    output_model = (
        "./artifacts/experiments/eos_8/sentence_Llama-3.2-3B_ft_4k_full_num_eos_tokens_8_frankenstein_from_4/checkpoint-8000"
    )

    # insert_position = 2
    # donor_gist_tokens_ids = [3, 4, 5, 6]

    donor_model, donor_tokenizer = load_model_from_checkpoint(donor_model)  # type: ignore[assignment]
    base_model, base_tokenizer = load_model_from_checkpoint(base_model)  # type: ignore[assignment]

    base_model.model.embed_tokens = donor_model.model.embed_tokens
    base_model.lm_head = donor_model.lm_head
    base_model.config = donor_model.config

    base_model.save_pretrained(output_model)
    donor_tokenizer.save_pretrained(output_model)
    print("Saved to ", output_model)

    # donor_gist_embeddings = []
    # token_embeddings = donor_model.model.embed_tokens.weight.clone() # type: ignore[attr-defined]

    # for eos_token_id in donor_model.config.end_of_sentence_token_ids:
    #     donor_gist_embeddings.append(token_embeddings[eos_token_id].detach())  # type: ignore[attr-defined]

    # base_token_embeddings_orig = base_model.model.embed_tokens.weight.clone() # type: ignore[attr-defined]
