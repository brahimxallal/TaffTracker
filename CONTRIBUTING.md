# Contributing

Thanks for your interest in the Vision Gimbal Tracker. This project targets
low-latency real-time operation, so changes must preserve or improve the
existing performance envelope (~59 fps, ~10 ms PC-side latency).

## Setup

1. Python 3.12 exactly (see `.python-version`).
2. `pip install -r requirements-dev.txt`
3. `pre-commit install` to enable format/lint hooks on every commit.

Optional for GPU-dependent features (TensorRT inference, real-time run):
install `torch` and `tensorrt` wheels matching your CUDA version separately.
The test suite itself runs CPU-only.

## Running tests

```bash
pytest                   # full suite
pytest -m unit           # fast unit layer
pytest -m integration    # cross-process tests
pytest -m "not perf"     # what CI runs
pytest --cov=src         # with coverage
```

## Commit convention

[Conventional Commits](https://www.conventionalcommits.org/): `<type>(<scope>): <subject>`

Types: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`, `ci`, `build`.

Examples:
- `fix(tracking): clamp Kalman velocity before prediction`
- `perf(inference): hoist protocol imports out of hot loop`
- `docs: add laser-boresight calibration walkthrough`

## Pull request checklist

- [ ] `pre-commit run --all-files` passes
- [ ] `pytest -m "not perf"` passes locally
- [ ] New logic has at least one test
- [ ] No secrets, weights (`*.pt`), engines (`*.engine`), or large binaries staged
- [ ] `git status` reviewed before `git add`
- [ ] Relevant section of `AGENTS.md` updated if architecture shifts

## Architecture guardrails

These invariants must not be broken without explicit discussion:

- Three-process pipeline boundary (Capture / Inference / Output).
- Lock-free `SharedRingBuffer` (no extra copies on the hot path).
- Fire-and-forget output transport (no blocking acks on the control path).
- Frozen-dataclass config (`src/config.py`) as the single source of runtime tuning.
- Kalman units: `vx`, `vy` are in **pixels/second**, not pixels/frame.

See `AGENTS.md` for the full list.

## Reporting a security issue

See [SECURITY.md](SECURITY.md). Please do not open public issues for vulnerabilities.
