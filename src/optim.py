import math

import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    loss = torch.logsumexp(logits, dim=-1) - target_logits
    return loss.mean()


def _validate_adamw_hyperparameters(lr, betas, eps, weight_decay):
    if lr < 0:
        raise ValueError("lr must be non-negative")
    if eps < 0:
        raise ValueError("eps must be non-negative")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
        raise ValueError("betas must each lie in [0, 1)")


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        _validate_adamw_hyperparameters(lr, betas, eps, weight_decay)
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        for group in self.param_groups:
            _validate_adamw_hyperparameters(group["lr"], group["betas"], group["eps"], group["weight_decay"])

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            _validate_adamw_hyperparameters(lr, (beta1, beta2), eps, weight_decay)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                with torch.no_grad():
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    bias_correction1 = 1 - beta1**t
                    bias_correction2 = 1 - beta2**t
                    m_hat = exp_avg / bias_correction1
                    v_hat = exp_avg_sq / bias_correction2
                    p.mul_(1 - lr * weight_decay)
                    p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

        return loss


def get_lr_cosine_schedule(step: int, learning_rate_max: float, learning_rate_min: float, warmup_steps: int, cosine_steps: int):
    if not isinstance(step, int) or isinstance(step, bool):
        raise TypeError("step must be an integer")
    if step < 0:
        raise ValueError("step must be non-negative")
    if not (0 <= learning_rate_min <= learning_rate_max):
        raise ValueError("learning_rate_min must lie in [0, learning_rate_max]")
    if warmup_steps < 0 or warmup_steps >= cosine_steps:
        raise ValueError("warmup_steps must be non-negative and less than cosine_steps")

    if step < warmup_steps:
        return step / warmup_steps * learning_rate_max

    if step <= cosine_steps:
        progress = (step - warmup_steps) / (cosine_steps - warmup_steps)
        return learning_rate_min + 0.5 * (1 + math.cos(math.pi * progress)) * (learning_rate_max - learning_rate_min)

    return learning_rate_min


def gradient_clipping(parameters, max_l2_norm: float):
    if not math.isfinite(max_l2_norm) or max_l2_norm <= 0:
        raise ValueError("max_l2_norm must be positive")
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0

    accumulator_dtype = torch.float64 if any(g.dtype == torch.float64 for g in grads) else torch.float32
    total_squared = torch.zeros((), device=grads[0].device, dtype=accumulator_dtype)
    for grad in grads:
        total_squared += grad.detach().to(device=grads[0].device, dtype=accumulator_dtype).square().sum()
    total_norm = torch.sqrt(total_squared)
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm.item() + 1e-6)
        for g in grads:
            g.mul_(scale)

    return total_norm.item()
