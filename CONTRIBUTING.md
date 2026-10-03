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

CI also runs `ruff check .`, `black --check .` and a non-blocking `mypy src`.
`requirements-dev.txt` installs the same pinned versions (`requirements-lint.txt`),
and `pre-commit run --all-files` runs the same ruff and black. mypy is a manual
hook: `pre-commit run --hook-stage manual mypy --all-files`.
The firmware host tests (`tests/test_firmware_host_harness.py`) need gcc or
clang with its sanitizer runtime. They skip themselves on a machine without one
and fail when `CI=true`, so CI cannot lose them silently.

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
- [ ] Firmware change? The host tests pass and the change was bench-tested on hardware (say so in the PR)

## Architecture guardrails

These invariants must not be broken without explicit discussion:

- Three-process pipeline boundary (Capture / Inference / Output).
- Lock-free `SharedRingBuffer` (no extra copies on the hot path).
- Fire-and-forget output transport (no blocking acks on the control path).
- Frozen-dataclass config (`src/config.py`) as the single source of runtime tuning.
- Kalman units: `vx`, `vy` are in **pixels/second**, not pixels/frame.
- Fail-safes stay: the firmware watchdog (coast / hold / home), the relay release when the host goes silent, and bounds-checked frame parsing.
- `relay.auto_fire_enabled` stays `false` in the committed `config.yaml`.

See `AGENTS.md` for the full list.

## Reporting a security issue

See [SECURITY.md](SECURITY.md). Please do not open public issues for vulnerabilities.
