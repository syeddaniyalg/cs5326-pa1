# CS 5326 PA1: The Modern Transformer LM

This repository implements a decoder-only Transformer language model using
PyTorch tensor operations. It includes training on TinyStories, validation,
text generation, and an analysis of the results.

The model uses pre-RMSNorm, adjacent-pair RoPE, SwiGLU, grouped-query attention,
and bias-free projections. Input embeddings and the output projection have
separate weights.

## Repository structure

```text
cs5326-pa1/
├── src/
│   ├── layers.py          # Linear, embedding, RMSNorm, and SwiGLU layers
│   ├── attention.py       # Softmax, RoPE, and grouped-query attention
│   ├── model.py           # Transformer block and language model
│   ├── optim.py           # Loss, AdamW, learning rate schedule, and clipping
│   ├── data.py            # Token loading, batch sampling, and checkpoints
│   ├── train.py           # Command-line training
│   └── generate.py        # Temperature and top-p sampling
├── tests/
│   ├── adapters.py        # Connects the implementation to the tests
│   ├── test_*.py          # Public tests
│   └── _snapshots/        # Reference outputs
├── report_assets/
│   ├── training_loss.png
│   ├── ablation_lr_2e4.png
│   ├── ablation_lr_3e4.png
│   ├── ablation_lr_4e4.png
│   └── sample_generated_text.txt
├── train.ipynb            # Kaggle setup and short learning-rate runs
├── REPORT.md              # Training results and decoding analysis
├── PA1.pdf                # Assignment manual
├── final_model.pt         # Exported model weights, stored locally
├── make_submission.sh
├── pyproject.toml
└── uv.lock
```

Model weights, checkpoints, and token streams are ignored by Git. A fresh clone
does not include `final_model.pt`; generate it with the full training command
below or copy it from the completed training run.

## Setup and tests

Use Python 3.11 or later and `uv`. From the repository root, run:

```bash
uv sync --frozen
uv run pytest
```

The environment includes PyTorch, NumPy, tokenizers, Hugging Face Hub,
matplotlib, and pytest. The tests use synthetic data, so the TinyStories files
are not needed to run them. Individual components can be tested separately:

```bash
uv run pytest tests/test_attention.py
uv run pytest tests/test_checkpoint.py
```

`tests/adapters.py` provides the interface used by the tests. Keep the public
test files unchanged. The assignment requirements are described in `PA1.pdf`.

## Model configuration

| Setting | Value |
|---|---:|
| Vocabulary size | 8,192 |
| Context length | 256 |
| Model width | 512 |
| Transformer blocks | 4 |
| Query heads | 16 |
| Key/value heads | 4 |
| SwiGLU hidden width | 1,344 |
| RoPE base | 10,000 |
| Parameters | 19,272,192 |

## Data

Download the supplied tokenizer and pretokenized TinyStories streams:

```bash
uv run hf download alooboii/pa1-tinystories metadata.json tokenizer/tokenizer.json data/train.bin data/validation.bin --repo-type dataset --local-dir data/tinystories
```

The training stream contains 466,876,982 tokens and the validation stream contains
4,692,376 tokens. The files contain little-endian `uint16` token IDs and are opened
with `numpy.memmap`. Training samples random windows with replacement and shifts
targets forward by one token. The tokenizer is used to encode generation prompts
and decode outputs; training reads the supplied token IDs directly.

## Training with the notebook

Open `train.ipynb` in Kaggle and enable a GPU and internet access. The first cell
clones the repository into `/kaggle/working/cs5326-pa1`. The second installs
Hugging Face Hub and tokenizers and downloads the course data into
`data/tinystories/` within that checkout. The notebook assumes its remaining
Python dependencies are available in the runtime. The setup commands above
provide the complete project environment when using `uv`.

The final cell launches two 1,000-update experiments with peak learning rates
of 3e-4 and 4e-4. Both use a microbatch size of 32, eight accumulation steps,
200 warmup steps, a cosine endpoint of 9,999, seed 42, and validation every
100 updates over 20 batches. Their checkpoints and plots have separate paths.
Both processes use the default CUDA device and run concurrently, so they share
GPU memory. Remove the background execution and run the commands sequentially
if the device cannot fit both jobs.

The 2e-4 experiment is represented by its saved plot, but its command
is not included in the notebook. Use the command-line trainer below for the
full 10,000-update run, final evaluation, sample generation, and model export.

## Command-line training

View all available settings with:

```bash
uv run python -m src.train --help
```

Running `uv run python -m src.train` uses the recommended starting configuration:
a microbatch size of 16, 16 accumulation steps, peak learning rate 3e-4, minimum
learning rate 3e-5, and validation every 200 updates. The reported experiment uses
the settings in the following command instead.

### Full run with the selected configuration

After downloading the data, run:

```bash
uv run python -m src.train --preflight --finalize --device cuda --num_steps 10000 --batch_size 32 --gradient_accumulation_steps 8 --learning_rate_max 4e-4 --learning_rate_min 4e-5 --warmup_steps 200 --cosine_steps 9999 --eval_interval 500 --log_interval 50 --checkpoint_interval 500 --num_validation_batches 20 --seed 42 --checkpoint_path checkpoints/final_checkpoint.pt
```

`--preflight` first checks whether a small model can overfit a fixed minibatch,
validates inference-mode evaluation, and checks that evaluation restores the
previous model mode. The full training run starts after these checks pass.

The effective batch contains 256 sequences of 256 tokens, giving 65,536 sampled
token positions per update and 655,360,000 across 10,000 updates. On CUDA, training
uses FP16 autocast with gradient scaling, FP32 parameters, and cross-entropy from
FP32 logits. Gradients are unscaled before clipping. Skipped scaler updates do
not advance the optimizer-update count.

Training metrics are printed every 50 updates. Validation and checkpointing run
every 500 updates and at the final update. Checkpoints contain model and optimizer
state, the gradient scaler, both sampling generators, the next update, and the
training history. The loss plot is saved to `report_assets/training_loss.png`.

### Resume or finalize a completed run

To resume the selected configuration, use:

```bash
uv run python -m src.train --resume --finalize --device cuda --num_steps 10000 --batch_size 32 --gradient_accumulation_steps 8 --learning_rate_max 4e-4 --learning_rate_min 4e-5 --warmup_steps 200 --cosine_steps 9999 --eval_interval 500 --log_interval 50 --checkpoint_interval 500 --num_validation_batches 20 --seed 42 --checkpoint_path checkpoints/final_checkpoint.pt
```

The checkpoint must exist. Keep the model, batching, schedule, and validation
settings consistent with the original run; these command-line settings are not
all stored in the checkpoint. If the checkpoint already contains 10,000 completed
updates, the command proceeds directly to final evaluation and export.

`--finalize` evaluates 100 validation batches of 16 sequences of length 256 with
a fresh generator seeded with 42. It prints mean cross-entropy and its exponential
as perplexity, exports `final_model.pt`, and generates the ten decoding examples.
The export is a plain CPU state dictionary with floating-point tensors converted
to FP16. Full optimizer checkpoints remain separate from this model-only file.

Generated text is saved to `report_assets/sample_generated_text.txt`. Use
`--plot_path`, `--generation_output_path`, and `--final_model_path` to choose
alternative output paths. The trainer stores history inside checkpoints and
prints metrics to the console; it does not create separate JSON result files.

## Recorded results

The final run used the supplied course streams, the fixed 19,272,192-parameter
architecture, and 10,000 optimizer updates. The selected learning-rate schedule
peaked at 4e-4 and ended at 4e-5.

### Short learning-rate comparisons

| Peak learning rate | Minimum learning rate | Validation loss at 1,000 updates |
|---|---|---:|
| 2e-4 | 2e-5 | 2.490369 |
| 3e-4 | 3e-5 | 2.314678 |
| 4e-4 | 4e-5 | 2.198591 |

The 4e-4 run improved on 3e-4 at every short-run validation checkpoint. This
controlled comparison provides the main basis for the selection.

### Final evaluation

| Metric | Result |
|---|---:|
| Training loss at update 10,000 | 1.6267 nats/token |
| Training-time validation loss at update 10,000 | 1.6497 nats/token |
| Standardized validation cross-entropy | 1.635051 nats/token |
| Standardized validation perplexity | 5.129721 |

The standardized evaluation used a fresh generator seeded with 42 and 100 batches
of 16 sequences of length 256. Perplexity is computed by exponentiating the mean
cross-entropy. These metrics were measured before converting weights to FP16.
The local exported weights match the FP16 conversion of the final checkpoint;
a separate evaluation after reloading the export was not performed.

### Decoding experiments

Both prompts use all five settings below:

- `Once upon a time`
- `Mia found a shiny red box in the park.`

| Temperature | Top-p |
|---:|---:|
| 0.7 | 0.9 |
| 1.0 | 0.9 |
| 1.2 | 0.9 |
| 1.0 | 0.7 |
| 1.0 | 1.0 |

Each experiment resets the sampling seed to 123 and generates up to 160 new
tokens, stopping earlier on `<|endoftext|>`. Temperature 0.7 with top-p 0.9 gives
the most readable pair of examples, although object changes and illogical
dialogue remain. Higher temperature and unrestricted sampling produce more
unusual details and weaker continuity in these samples.

[REPORT.md](REPORT.md) contains all ten prompts, parameter settings, complete
outputs, and their interpretation, together with training figures and the
limitations of the comparisons.

## Submission

Place `final_model.pt` in the repository root and keep report figures under
`report_assets/`. From the repository root, run:

```bash
bash make_submission.sh
```

The script requires Bash, `uv`, and `unzip`. It uses Python's ZIP support when
`zip` is unavailable. It runs the public tests,
checks the required artifacts, and creates `submission.zip`. Test failures do
not stop packaging, but missing or invalid required artifacts do.

The archive contains:

- `src/`
- `tests/adapters.py`
- `REPORT.md`
- `report_assets/`, including the plot and saved text examples
- `final_model.pt`

The notebook, README, console logs, token streams, and full training checkpoints
are excluded. Rename the archive to `<roll_number_pa1>.zip`,
replacing `<roll_number>` with your roll number, before uploading it to the LMS.
