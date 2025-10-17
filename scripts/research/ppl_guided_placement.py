import os

import matplotlib.pyplot as plt
import torch
from sentence_attention.evaluation.my_recall import generate_random_sample
from sentence_attention.models.checkpoint import load_model_from_checkpoint
from torch.nn import functional as F


def visualize_tokens_logits(tokens_logits, tokens, figsize=(18, 6), output_path="/tmp/tokens_logits.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=figsize)
    positions = list(range(len(tokens)))
    plt.bar(positions, tokens_logits)
    plt.xticks(positions, tokens, rotation=90)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved to {output_path}")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="/tmp/tokens_logits.png")
    args = parser.parse_args()

    model_name = args.model_name

    model, tokenizer = load_model_from_checkpoint(model_name)

    with torch.no_grad():
        model.to("cuda")
        model.eval()

        text = generate_random_sample(num_examples=3)
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        outputs = model(**inputs)

        log_probs = F.softmax(outputs.logits, dim=-1)
        labels = inputs.input_ids[:, 1:]

        tokens_logits = torch.gather(log_probs[:, :-1, :], dim=-1, index=labels.unsqueeze(-1))

        tokens_logits = tokens_logits[0, :, 0].cpu().numpy()
        labels = labels[0, :].cpu().numpy()

        # Convert token ids to readable token strings for x-axis labels
        token_texts = tokenizer.convert_ids_to_tokens(labels.tolist())
        token_texts = [t.removeprefix("Ġ") for t in token_texts]
        token_texts = [t.replace("Ċ", "\\n") for t in token_texts]

        # Visualize tokens_logits with proper x-axis positions and labels
        visualize_tokens_logits(tokens_logits, tokens=token_texts, output_path=args.output_path)

        breakpoint()
