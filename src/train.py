import argparse
import os

import torch
from tqdm import tqdm

from src.data import get_batch, load_checkpoint, load_token_array, save_checkpoint
from src.model import TransformerLM
from src.optim import AdamW, cross_entropy, get_lr_cosine_schedule, gradient_clipping


def evaluate_validation(model, val_tokens, batch_size, sequence_length, device, generator, num_batches):
    was_training = model.training
    model.eval()
    losses = []
    with torch.inference_mode():
        for _ in range(num_batches):
            x, y = get_batch(val_tokens, batch_size, sequence_length, device, generator)
            losses.append(cross_entropy(model(x), y).item())
    model.train(was_training)
    return sum(losses) / len(losses)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/tinystories/data/train.bin")
    parser.add_argument("--val_path", default="data/tinystories/data/validation.bin")
    parser.add_argument("--checkpoint_path", default="checkpoints/checkpoint.pt")
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--n_q_heads", type=int, default=16)
    parser.add_argument("--n_kv_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--norm_eps", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--learning_rate_max", type=float, default=3e-4)
    parser.add_argument("--learning_rate_min", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--cosine_steps", type=int, default=9999)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--checkpoint_interval", type=int, default=500)
    parser.add_argument("--num_validation_batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device
    print(f"requested device={device} cuda_available={torch.cuda.is_available()} torch={torch.__version__}", flush=True)

    train_tokens = load_token_array(args.train_path)
    val_tokens = load_token_array(args.val_path)

    model = TransformerLM(
        args.vocab_size, args.context_length, args.d_model, args.num_layers,
        args.n_q_heads, args.n_kv_heads, args.d_ff, args.rope_theta,
        norm_eps=args.norm_eps, device=device,
    )
    print(f"model.lm_head.weight.device={model.lm_head.weight.device} num_params={sum(p.numel() for p in model.parameters())}", flush=True)
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate_max,
        betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.weight_decay,
    )

    train_generator = torch.Generator().manual_seed(args.seed)
    val_generator = torch.Generator().manual_seed(args.seed + 1)

    next_step = 0
    if args.resume and os.path.exists(args.checkpoint_path):
        next_step = load_checkpoint(args.checkpoint_path, model, optimizer, train_generator, val_generator)

    for step in tqdm(range(next_step, args.num_steps), mininterval=5.0):
        model.train()
        lr = get_lr_cosine_schedule(step, args.learning_rate_max, args.learning_rate_min, args.warmup_steps, args.cosine_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad()
        train_loss = 0.0
        for _ in range(args.gradient_accumulation_steps):
            x, y = get_batch(train_tokens, args.batch_size, args.context_length, device, train_generator)
            microbatch_loss = cross_entropy(model(x), y)
            (microbatch_loss / args.gradient_accumulation_steps).backward()
            train_loss += microbatch_loss.item()
        train_loss /= args.gradient_accumulation_steps

        grad_norm = gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        completed_steps = step + 1
        final_step = completed_steps == args.num_steps
        should_validate = final_step or completed_steps % args.eval_interval == 0
        should_log = should_validate or completed_steps % args.log_interval == 0
        should_checkpoint = final_step or completed_steps % args.checkpoint_interval == 0

        if should_validate:
            validation_loss = evaluate_validation(model, val_tokens, args.batch_size, args.context_length, device, val_generator, args.num_validation_batches)

        if should_log:
            message = f"step {completed_steps} lr {lr:.6f} train_loss {train_loss:.4f} grad_norm {grad_norm:.4f}"
            if should_validate:
                message += f" val_loss {validation_loss:.4f}"
            print(message)

        if should_checkpoint:
            checkpoint_dir = os.path.dirname(args.checkpoint_path)
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
            save_checkpoint(model, optimizer, completed_steps, train_generator, val_generator, args.checkpoint_path)


if __name__ == "__main__":
    main()
