import os

import numpy as np
import torch


def load_token_array(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if os.path.getsize(path) % 2 != 0:
        raise ValueError("token stream must contain a whole number of uint16 IDs")
    return np.memmap(path, mode="r", dtype=np.dtype("<u2"))


def get_batch(dataset, batch_size: int, sequence_length: int, device, generator):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if len(dataset) <= sequence_length:
        raise ValueError("dataset must contain at least one window of sequence_length + 1 tokens")

    starts = torch.randint(0, len(dataset) - sequence_length, (batch_size,), generator=generator)
    x_rows = [dataset[s : s + sequence_length].astype(np.int64) for s in starts.tolist()]
    y_rows = [dataset[s + 1 : s + sequence_length + 1].astype(np.int64) for s in starts.tolist()]
    x = torch.from_numpy(np.stack(x_rows))
    y = torch.from_numpy(np.stack(y_rows))
    return x.to(device), y.to(device)


def save_checkpoint(model, optimizer, next_step: int, train_generator, val_generator, out):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "next_step": next_step,
        "train_generator": train_generator.get_state(),
        "val_generator": val_generator.get_state(),
    }, out)


def load_checkpoint(src, model, optimizer, train_generator, val_generator):
    checkpoint = torch.load(src, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    train_generator.set_state(checkpoint["train_generator"])
    val_generator.set_state(checkpoint["val_generator"])
    return checkpoint["next_step"]
