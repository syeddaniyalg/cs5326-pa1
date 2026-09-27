import torch

from src.attention import softmax


def generate(model, prompt_ids: torch.Tensor, max_new_tokens: int, context_length: int, temperature=1.0, top_p=1.0, eot_token_id=None, generator=None):
    assert prompt_ids.numel() > 0
    assert max_new_tokens >= 0
    assert context_length > 0 and context_length == model.context_length
    assert temperature > 0
    assert 0 < top_p <= 1

    if max_new_tokens == 0:
        return prompt_ids

    was_training = model.training
    model.eval()
    sequence = prompt_ids

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            context = sequence[-context_length:].unsqueeze(0)
            logits = model(context)[0, -1]
            probs = softmax(logits / temperature, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            nucleus_size = min((cumulative < top_p).sum().item() + 1, sorted_probs.numel())
            sorted_probs = sorted_probs[:nucleus_size]
            sorted_indices = sorted_indices[:nucleus_size]
            sorted_probs = sorted_probs / sorted_probs.sum()

            sampled_rank = torch.multinomial(sorted_probs, 1, generator=generator)
            next_token = sorted_indices[sampled_rank]
            sequence = torch.cat([sequence, next_token])

            if eot_token_id is not None and next_token.item() == eot_token_id:
                break

    model.train(was_training)
    return sequence
