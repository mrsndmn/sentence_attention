import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def visualize_tokens_logits(tokens_logits, tokens):
    plt.figure(figsize=(10, 10))
    plt.bar(range(len(tokens)), tokens_logits)
    plt.xticks(tokens, rotation=90)
    plt.savefig("/tmp/tokens_logits.png")
    plt.close()
    print("Saved to /tmp/tokens_logits.png")


if __name__ == "__main__":

    model_name = "unsloth/Llama-3.2-3B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    with torch.no_grad():
        model.to("cuda")
        model.eval()

        text = "Hello, how are you? Thank you, i'm fine!"
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        outputs = model(**inputs)

        log_probs = F.log_softmax(outputs.logits, dim=-1)
        labels = inputs.input_ids[:, 1:]

        tokens_logits = torch.gather(log_probs[:, :-1, :], dim=-1, index=labels.unsqueeze(-1))

        tokens_logits = tokens_logits[0, :, 0].cpu().numpy()
        labels = labels[0, :].cpu().numpy()

        # Visualize tokens_logits
        visualize_tokens_logits(tokens_logits, tokens=labels)
