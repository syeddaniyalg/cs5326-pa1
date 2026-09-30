#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d "src" || ! -f "tests/adapters.py" || ! -f "REPORT.md" ]]; then
  echo "Run this script from the assignment repository root." >&2
  exit 1
fi

if [[ ! -f "final_model.pt" ]]; then
  echo "Missing required submission file: final_model.pt" >&2
  exit 1
fi
if [[ ! -s "final_model.pt" ]]; then
  echo "final_model.pt must not be empty." >&2
  exit 1
fi
if ! find src -type f -name '*.py' -print -quit | grep -q .; then
  echo "The src/ directory does not contain a Python implementation." >&2
  exit 1
fi
if [[ ! -d "report_assets" ]] || ! find report_assets -type f ! -name '.gitkeep' ! -name '.DS_Store' -print -quit | grep -q .; then
  echo "report_assets/ must contain the figures referenced by REPORT.md." >&2
  exit 1
fi
if grep -q 'TODO: connect your implementation\|NotImplementedError' tests/adapters.py; then
  echo "tests/adapters.py still contains an unfinished adapter." >&2
  exit 1
fi

set +e
uv run pytest -q
test_status=$?
set -e
if [[ "$test_status" -ne 0 ]]; then
  echo "Public tests failed; packaging will continue so partial work can be submitted." >&2
fi

uv run python - <<'PY'
from pathlib import Path

import torch

model_path = Path("final_model.pt")
state = torch.load(model_path, weights_only=True)
if not isinstance(state, dict) or not state:
    raise SystemExit("final_model.pt must contain a non-empty state dictionary")
if not all(
    isinstance(name, str) and isinstance(value, torch.Tensor)
    for name, value in state.items()
):
    raise SystemExit("final_model.pt must map parameter names directly to tensors")
if sum(tensor.numel() for tensor in state.values()) != 19_272_192:
    raise SystemExit("final_model.pt does not contain exactly 19,272,192 values")
for name, tensor in state.items():
    if tensor.device.type != "cpu":
        raise SystemExit(f"{name} is not stored on CPU")
    if tensor.is_floating_point():
        if tensor.dtype != torch.float16:
            raise SystemExit(f"{name} is not stored as float16")
        if not torch.isfinite(tensor).all():
            raise SystemExit(f"{name} contains a non-finite value")
PY

staging_dir="$(mktemp -d "${TMPDIR:-/tmp}/cs5326-pa1-submission.XXXXXX")"
trap 'rm -rf "$staging_dir"' EXIT

mkdir -p "$staging_dir/tests"
cp -R src "$staging_dir/src"
cp tests/adapters.py "$staging_dir/tests/adapters.py"
cp REPORT.md final_model.pt "$staging_dir/"
cp -R report_assets "$staging_dir/report_assets"

find "$staging_dir" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$staging_dir" -type f \( -name '*.pyc' -o -name '.DS_Store' -o -name '.gitkeep' \) -delete

archive="submission.zip"
rm -f "$archive"
if command -v zip >/dev/null 2>&1; then
  (cd "$staging_dir" && zip -qr "$OLDPWD/$archive" .)
else
  uv run python - "$staging_dir" "$archive" <<'PY'
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

staging_dir = Path(sys.argv[1])
with ZipFile(sys.argv[2], "w", compression=ZIP_DEFLATED) as archive:
    for path in sorted(staging_dir.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(staging_dir).as_posix())
PY
fi

echo "Created $archive."
unzip -l "$archive"
echo
echo "Rename submission.zip to <roll_number_pa1>.zip, replacing <roll_number> with your roll number, before uploading it to the LMS."
