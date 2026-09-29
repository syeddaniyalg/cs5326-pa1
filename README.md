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
│   └── sample_generated_text.txt
├── train.ipynb            # Kaggle training and evaluation notebook
├── REPORT.md              # Training results and decoding analysis
├── PA1.pdf                # Assignment manual
├── final_model.pt         # Exported model weights, stored locally
├── make_submission.sh
├── pyproject.toml
└── uv.lock
```

Model weights, checkpoints, and token streams are ignored by Git. A fresh clone
does not include `final_model.pt`; generate it with the notebook or copy it from
the completed training run.

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

## Training with the notebook

Open `train.ipynb` in Kaggle and enable a GPU and internet access. Run the clone
cell once, then run the remaining cells in order. The notebook downloads the
supplied tokenizer and pretokenized training and validation streams to
`/kaggle/working/tinystories/`.

For a new run, leave `resume = False`. To resume, set it to `True` and place the
checkpoint at `/kaggle/working/official_checkpoint.pt` before running the model
and resume cells.

Training uses a microbatch size of 32, eight accumulation steps, and 10,000
optimizer updates. Each update uses 65,536 token positions. The notebook uses
FP16 autocast with gradient scaling, prints training metrics after each update,
and evaluates and saves a checkpoint every 500 updates.

After training, it plots losses, evaluates the final model, compares generation
settings, and exports `final_model.pt`. The plot and model weights are saved in
the cloned repository. Training history, final metrics, and generated samples
are also saved as JSON files under `/kaggle/working/`.

## Data and command-line training

The notebook and command-line trainer use the supplied course streams. To
download them manually, run:

```bash
uv run hf download alooboii/pa1-tinystories metadata.json tokenizer/tokenizer.json data/train.bin data/validation.bin --repo-type dataset --local-dir data/tinystories
```

The files contain little-endian `uint16` token IDs and are opened with
`numpy.memmap`. The command-line trainer uses these paths by default:

```bash
uv run python -m src.train
```

View the available settings or resume from the default checkpoint with:

```bash
uv run python -m src.train --help
uv run python -m src.train --resume
```

On CUDA, the command-line trainer automatically uses FP16 autocast and gradient
scaling. Run the small-model fixed-minibatch check before a full run with:

```bash
uv run python -m src.train --preflight
```

Add `--finalize` to run the standardized 100-batch evaluation, print perplexity,
generate samples at several temperature and top-p settings, and export the model
to `final_model.pt`:

```bash
uv run python -m src.train --preflight --finalize
```

Training history is stored in each checkpoint. The CLI saves the training and
validation loss plot to `report_assets/training_loss.png` and saves generated
samples to `report_assets/sample_generated_text.txt`. Use `--plot_path` or
`--generation_output_path` to choose different locations.

The full checkpoint includes the gradient scaler state. If `--resume` is used,
the checkpoint must exist. A completed checkpoint can be evaluated and exported
without further updates by using `--resume --finalize` with the same
`--num_steps` value.

## Recorded results

The existing model and report come from an earlier 10,000-update run on a Tesla
T4. Final evaluation used a fresh generator seeded with 42 and 100 batches of 16
sequences of length 256.

| Metric | Result |
|---|---:|
| Validation cross-entropy | 1.482880 nats/token |
| Validation perplexity | 4.405615 |

Perplexity is computed by exponentiating the mean cross-entropy. These results
apply to the earlier retokenized Kaggle validation data and were measured before
exporting the model weights to FP16. Run the updated notebook to produce results
for the supplied pretokenized streams.

Generation compares temperatures from 0.1 to 2.0 and top-p values of 0.7, 0.9,
and 1.0 across two prompts. The saved examples show readable stories at moderate
temperatures and substantial grammatical and logical errors at high temperatures.
See `REPORT.md` for the analysis, training plot, and limitations of the experiments.

## Submission

Place `final_model.pt` in the repository root and keep report figures under
`report_assets/`. From the repository root, run:

```bash
bash make_submission.sh
```

The script requires Bash, `uv`, `zip`, and `unzip`. It runs the public tests,
checks the required artifacts, and creates `submission.zip`. Test failures do
not stop packaging, but missing or invalid required artifacts do.

The archive contains:

- `src/`
- `tests/adapters.py`
- `REPORT.md`
- `report_assets/`, including the plot and saved text examples
- `final_model.pt`

The notebook, README, JSON logs outside `report_assets/`, token streams, and
training checkpoints are excluded. Rename the archive to `<roll_number_pa1>.zip`,
replacing `<roll_number>` with your roll number, before uploading it to the LMS.
