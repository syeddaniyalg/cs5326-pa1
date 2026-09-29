import argparse
import math
import os

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from tokenizers import Tokenizer

from src.data import get_batch, load_checkpoint, load_token_array, save_checkpoint
from src.generate import generate
from src.model import TransformerLM
from src.optim import AdamW, cross_entropy, get_lr_cosine_schedule, gradient_clipping


def evaluate_validation(model, val_tokens, batch_size, sequence_length, device, generator, num_batches):
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    was_training = model.training
    model.eval()
    losses = []
    device_type = torch.device(device).type
    try:
        with torch.inference_mode():
            for _ in range(num_batches):
                x, y = get_batch(val_tokens, batch_size, sequence_length, device, generator)
                with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=device_type == "cuda"):
                    logits = model(x)
                losses.append(cross_entropy(logits.float(), y).item())
    finally:
        model.train(was_training)
    return sum(losses) / len(losses)


def run_preflight(train_tokens, val_tokens, vocab_size, device, seed, num_steps):
    context_length = 64
    torch.manual_seed(seed)
    model = TransformerLM(vocab_size, context_length, 64, 2, 4, 2, 192, 10000.0, device=device)
    optimizer = AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    x, y = get_batch(train_tokens, 2, context_length, device, generator)
    losses = []

    for _ in tqdm(range(num_steps), desc="preflight", unit="step"):
        model.train()
        optimizer.zero_grad()
        loss = cross_entropy(model(x), y)
        loss.backward()
        gradient_clipping(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    with torch.inference_mode():
        final_loss = cross_entropy(model(x), y).item()

    validation_states = []

    def record_state(module, inputs):
        validation_states.append((torch.is_inference_mode_enabled(), module.training))

    handle = model.register_forward_pre_hook(record_state)
    model.train()
    previous_mode = model.training
    val_generator = torch.Generator().manual_seed(seed + 1)
    validation_loss = evaluate_validation(model, val_tokens, 2, context_length, device, val_generator, 2)
    handle.remove()

    if final_loss >= losses[0] * 0.5:
        raise RuntimeError("preflight model did not sharply overfit the fixed minibatch")
    if not validation_states or not all(inference and not training for inference, training in validation_states):
        raise RuntimeError("validation did not run in inference and evaluation mode")
    if model.training != previous_mode:
        raise RuntimeError("validation did not restore the previous model mode")

    print(f"preflight initial_loss={losses[0]:.4f} final_loss={final_loss:.4f} validation_loss={validation_loss:.4f}")


def save_training_plot(history, path):
    if not history:
        return
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    validation = [row for row in history if "validation_loss" in row]
    plt.figure(figsize=(9, 5))
    plt.plot([row["step"] for row in history], [row["train_loss"] for row in history], label="Training")
    if validation:
        plt.plot([row["step"] for row in validation], [row["validation_loss"] for row in validation], "o-", label="Validation")
    plt.xlabel("Optimizer update")
    plt.ylabel("Cross-entropy")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def finalize(model, val_tokens, args, device):
    if args.context_length < 256:
        raise ValueError("standardized final evaluation requires context_length >= 256")

    final_generator = torch.Generator().manual_seed(42)
    final_loss = evaluate_validation(model, val_tokens, 16, 256, device, final_generator, 100)
    perplexity = math.exp(final_loss)
    print(f"final_validation_loss={final_loss:.6f} final_perplexity={perplexity:.6f}")

    state = {}
    for name, tensor in model.state_dict().items():
        tensor = tensor.detach().cpu()
        state[name] = tensor.half() if tensor.is_floating_point() else tensor

    output_dir = os.path.dirname(args.final_model_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(state, args.final_model_path)
    print(f"saved final model to {args.final_model_path}")

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    eot_token_id = tokenizer.token_to_id("<|endoftext|>")
    prompts = ["Once upon a time", "Mia found a shiny red box in the park."]
    settings = [(0.7, 0.9), (1.0, 0.9), (1.2, 0.9), (1.0, 0.7), (1.0, 1.0)]
    samples = []

    for prompt in prompts:
        prompt_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False).ids, device=device)
        for temperature, top_p in settings:
            generator = torch.Generator(device=device).manual_seed(123)
            output = generate(model, prompt_ids, args.generation_tokens, args.context_length, temperature, top_p, eot_token_id, generator)
            text = tokenizer.decode(output.tolist(), skip_special_tokens=False)
            sample = f"Prompt: {prompt}\nTemperature: {temperature}\nTop-p: {top_p}\n\n{text}"
            samples.append(sample)
            print(f"\n{sample}")

    output_dir = os.path.dirname(args.generation_output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.generation_output_path, "w", encoding="utf-8") as file:
        file.write("\n\n".join(samples))
    print(f"saved generated text to {args.generation_output_path}")


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
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight_steps", type=int, default=400)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--final_model_path", default="final_model.pt")
    parser.add_argument("--tokenizer_path", default="data/tinystories/tokenizer/tokenizer.json")
    parser.add_argument("--generation_tokens", type=int, default=160)
    parser.add_argument("--generation_output_path", default="report_assets/sample_generated_text.txt")
    parser.add_argument("--plot_path", default="report_assets/training_loss.png")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    for name in ("batch_size", "gradient_accumulation_steps", "num_steps", "eval_interval", "log_interval", "checkpoint_interval", "num_validation_batches", "preflight_steps", "generation_tokens"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")

    device = args.device
    print(f"requested device={device} cuda_available={torch.cuda.is_available()} torch={torch.__version__}", flush=True)

    train_tokens = load_token_array(args.train_path)
    val_tokens = load_token_array(args.val_path)

    if args.preflight:
        run_preflight(train_tokens, val_tokens, args.vocab_size, device, args.seed + 100, args.preflight_steps)

    torch.manual_seed(args.seed)
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
    device_type = torch.device(device).type
    use_amp = device_type == "cuda"
    scaler = torch.amp.GradScaler(device_type, enabled=use_amp)

    train_generator = torch.Generator().manual_seed(args.seed)
    val_generator = torch.Generator().manual_seed(args.seed + 1)
    history = []

    next_step = 0
    if args.resume:
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(args.checkpoint_path)
        next_step = load_checkpoint(args.checkpoint_path, model, optimizer, train_generator, val_generator, scaler, history)

    step = next_step
    with tqdm(total=args.num_steps, initial=next_step, mininterval=1.0, unit="update") as progress:
        while step < args.num_steps:
            model.train()
            lr = get_lr_cosine_schedule(step, args.learning_rate_max, args.learning_rate_min, args.warmup_steps, args.cosine_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad()
            train_loss = 0.0
            for _ in range(args.gradient_accumulation_steps):
                x, y = get_batch(train_tokens, args.batch_size, args.context_length, device, train_generator)
                with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
                    logits = model(x)
                microbatch_loss = cross_entropy(logits.float(), y)
                scaler.scale(microbatch_loss / args.gradient_accumulation_steps).backward()
                train_loss += microbatch_loss.item()
            train_loss /= args.gradient_accumulation_steps

            scaler.unscale_(optimizer)
            grad_norm = gradient_clipping(model.parameters(), args.max_grad_norm)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if use_amp and scaler.get_scale() < old_scale:
                continue

            completed_steps = step + 1
            final_step = completed_steps == args.num_steps
            should_validate = final_step or completed_steps % args.eval_interval == 0
            should_log = should_validate or completed_steps % args.log_interval == 0
            should_checkpoint = final_step or completed_steps % args.checkpoint_interval == 0

            if should_validate:
                validation_loss = evaluate_validation(model, val_tokens, args.batch_size, args.context_length, device, val_generator, args.num_validation_batches)

            row = {"step": completed_steps, "train_loss": train_loss, "lr": lr, "grad_norm": grad_norm}
            if should_validate:
                row["validation_loss"] = validation_loss
            history.append(row)

            if should_log:
                message = f"step {completed_steps} lr {lr:.6f} train_loss {train_loss:.4f} grad_norm {grad_norm:.4f}"
                if should_validate:
                    message += f" val_loss {validation_loss:.4f}"
                print(message)

            if should_checkpoint:
                checkpoint_dir = os.path.dirname(args.checkpoint_path)
                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                save_checkpoint(model, optimizer, completed_steps, train_generator, val_generator, args.checkpoint_path, scaler, history)
                save_training_plot(history, args.plot_path)

            step = completed_steps
            progress.update(1)

    save_training_plot(history, args.plot_path)

    if args.finalize:
        finalize(model, val_tokens, args, device)


if __name__ == "__main__":
    main()
