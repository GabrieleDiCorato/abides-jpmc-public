# Contributing to ABIDES

Thanks for your interest in contributing. This project is currently
solo-maintained on a best-effort basis but **welcomes co-maintainers
and external contributors** — open an issue if you'd like to take on
a larger role.

## Reporting bugs and requesting features

Use the GitHub issue tracker:
<https://github.com/GabrieleDiCorato/abides-ng/issues>

For security-sensitive reports, see [SECURITY.md](SECURITY.md) instead.

## Development setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/GabrieleDiCorato/abides-ng
cd abides-ng
uv sync --dev
uv run pre-commit install
```

This installs the workspace in editable mode and registers the
pre-commit hooks (black, ruff, mypy, nb-clean, basic file checks).

### Editable-install caveat

The kernel source lives in `abides-core/abides_core/` but is bundled
into the published `abides-ng` wheel via `force-include`. As a
result, an editable install of `abides-ng` copies a snapshot of
`abides_core` into `site-packages` rather than linking it.

**When you edit code in `abides-core/abides_core/`, refresh the
editable copy:**

```bash
uv sync --reinstall-package abides-ng
```

Tests under `abides-core/tests/` and `abides-markets/tests/` import
the installed copy, so this step is required before re-running them
after kernel changes. (We considered using `dev-mode-dirs` to avoid
the copy, but hatchling currently inherits `force-include` into the
editable target.)

## Pull request workflow

1. Fork the repo and create a branch off `main`.
2. Make focused commits — group related changes, separate unrelated
   ones.
3. Run `uv run pre-commit run --all-files` — all hooks must pass.
4. Run the relevant tests: `uv run pytest <path/to/test_file.py>`.
   Avoid running the full suite locally (it takes ~4 minutes); CI
   runs the full matrix.
5. Open a PR against `main`. Describe the change, its motivation,
   and any reproducibility implications.

### Commit messages

Use **past tense, imperative-free**:

- Good: `"Added MeanReversionAgent strategy parameter for half-life."`
- Good: `"Fixed off-by-one in latency-noise RNG draw."`
- Avoid: `"Add ..."`, `"Fix ..."`, `"WIP: ..."`

Don't reference internal task tracker IDs or PR numbers in the
subject line — those belong in the PR description.

### Coding standards

- **Integer cents for prices.** Never use floats internally.
  `$100.00 = 10_000`. Display only: `f"${price / 100:.2f}"`.
- **Event-driven.** Agents react via `wakeup()` and
  `receive_message()` only — no synchronous waits on market state.
- **Guard market data.** `self.mkt_open`, `self.known_bids[symbol]`,
  `L1DataMsg.bid`, etc. may be `None` or empty. Always guard.
- **Type hints required** on new public APIs. `mypy` runs in CI.
- **No new dependencies** without justification — discuss in the
  issue first.

See [docs/reference/llm-gotchas.md](docs/reference/llm-gotchas.md)
for the full list of safe patterns.

## Tests

- Live in `abides-core/tests/` (kernel) and `abides-markets/tests/`
  (markets, configs, agents).
- New behaviour requires a test. Bug fixes require a regression
  test that fails before the fix and passes after.
- **Don't add tests for scenarios that cannot realistically occur** —
  test observable behaviour and real edge cases only.
- Reproducibility-sensitive changes must include or update a test
  that locks the new behaviour (see
  `abides-core/tests/test_seed_replicability.py`).

## Documentation

- Architecture and design references live in `docs/reference/`.
- User-facing changes should update `CHANGELOG.md` under
  `Unreleased`.
- New public classes/methods need docstrings.
- Don't create new markdown files to document changes — update
  the existing reference docs and the changelog.

## License and copyright

By contributing, you agree that your contributions will be licensed
under the BSD-3-Clause License (see [LICENSE](LICENSE)). No CLA or
DCO sign-off is required.
