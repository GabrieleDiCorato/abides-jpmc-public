# Release Process

How releases of `abides-ng` (and eventually `abides-ng[gym]`) are produced and
published. This document is the operational reference for the
release workflow at
[.github/workflows/release.yml](../../.github/workflows/release.yml).

## Distribution overview

The monorepo defines two PyPI distributions, but **only `abides-ng`
is published in the v2.6.x line**:

- **`abides-ng`** — kernel + market microstructure. Source lives in
  `abides-markets/` (directory keeps its original name). Bundles
  `abides_core` via hatchling `force-include` from
  `abides-core/abides_core/`. **Published.**
- **`abides-gym`** — Gymnasium and RLlib adapters. Depends on
  `abides-ng==<same-version>`. **Deferred from v2.6.x** — the
  gym adapter has not been re-validated against the new
  `SimulationConfig` system and has no pytest coverage in this repo.
  It will ship as a separate `abides-gym` release once validated.
  Until then, install it from source: `pip install -e abides-gym/`.

The release is **wheel-only**: no sdist is published. Source builds
require cloning the repository (and are CI-tested via `uv build`).

## Versioning

Semantic versioning. Current line is `2.x`; v3.0 will land when the
`abides-core-v3-foundation` plan completes (see
[docs/active-plans/](../active-plans/) when active, or git history).

Reproducibility-breaking changes (anything that alters the output of
a seeded simulation bit-for-bit) require a **major-version bump** and
must be called out in `CHANGELOG.md` under `Breaking Changes`. See
[reproducibility.md](reproducibility.md).

### Tag format

- Stable release: `v2.6.0`, `v2.7.0`, `v3.0.0`
- Pre-release: `v2.7.0-rc1`, `v3.0.0-a1`, `v3.0.0-b2`

The leading `v` is the **git-tag convention** only; the
`pyproject.toml` `version` field is bare PEP 440 (`2.6.0`).

The release workflow strips the `v` and verifies that the tag
version matches `abides-markets/pyproject.toml` before publishing.

## One-time setup: PyPI Trusted Publishing

This must be done **once per project, per index** before the first
upload. Long-lived API tokens are not used.

### TestPyPI

1. Log in at <https://test.pypi.org/manage/account/publishing/>.
2. Add a new "pending publisher" for `abides-ng`:
   - Owner: `GabrieleDiCorato`
   - Repository: `abides-ng`
   - Workflow: `release.yml`
   - Environment: `testpypi`
3. After the first successful TestPyPI upload, the publisher
   becomes "active" and the project is owned by the trusted
   publisher.

   When `abides-gym` is later cleared for release, repeat with
   project name `abides-ng` (same wheel, `[gym]` extra bundled in).

### PyPI

Same steps at <https://pypi.org/manage/account/publishing/>, but use
environment name `pypi`.

### GitHub repository environments

Create two GitHub Environments at Settings → Environments:

- `testpypi` — no protection rules required.
- `pypi` — **add a deployment protection rule** that requires
  manual approval before any job in this environment runs. This is
  the safety brake before a real PyPI upload.

## Release workflow

The workflow at `.github/workflows/release.yml` runs on tag pushes:

| Tag pattern        | Builds wheels | Uploads to TestPyPI | Uploads to PyPI | GitHub Release |
|--------------------|---------------|---------------------|-----------------|----------------|
| `v2.6.0` (stable)  | yes           | no                  | yes             | yes            |
| `v2.6.0-rc1`       | yes           | yes                 | no              | no             |
| `workflow_dispatch`| yes           | yes (manual)        | no              | no             |

### Job graph

1. **build** — checks out, builds the `abides-ng` wheel, runs
   `twine check`, uploads the wheel as a workflow artifact. Verifies
   tag-vs-pyproject version match.
2. **publish-testpypi** — only on prerelease tags or manual dispatch.
   Uses OIDC trusted publishing.
3. **publish-pypi** — only on stable tags. Uses OIDC trusted
   publishing. Gated by the `pypi` environment's manual-approval
   rule.
4. **github-release** — only on stable tags, after `publish-pypi`
   succeeds. Extracts the matching CHANGELOG section, creates the
   GitHub Release, attaches the wheel.

## Cutting a release

### 1. Prepare on `main`

- Land all release content via PRs.
- Update `CHANGELOG.md`: move the `Unreleased` section to a new
  dated `Release vX.Y.Z` heading, add new `Unreleased` placeholder.
- Bump `version` in `abides-markets/pyproject.toml` (the `abides-ng` wheel;
  source dir is named `abides-markets/` but the distribution name is `abides-ng`).
  If/when `abides-gym` is cleared for release, bump it too and update its
  `abides-ng==<ver>` pin to match.
- Run `uv sync` so `uv.lock` reflects the new versions; commit.
- Run `uv run pre-commit run --all-files` until clean.

### 2. Local verification

```bash
rm -rf dist
(cd abides-markets && uv build --wheel)   # produces abides_ng-*.whl
uv run twine check dist/*
```

Optional sanity check: install both wheels in a fresh venv and run
a smoke simulation from `notebooks/`.

### 3. Pre-release dry-run (TestPyPI)

For the first release on a new line (e.g. v3.0.0) or anytime you
want a rehearsal:

```bash
git tag v2.6.0-rc1
git push origin v2.6.0-rc1
```

Workflow runs → uploads to TestPyPI. In a fresh venv:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            "abides-ng==2.6.0rc1"
abides --help
python -c "import abides_core, abides_markets; print('ok')"
```

If broken, delete the tag locally and on origin, fix, and try
`-rc2`:

```bash
git tag -d v2.6.0-rc1
git push origin :refs/tags/v2.6.0-rc1
```

### 4. Stable release

```bash
git tag v2.6.0
git push origin v2.6.0
```

Workflow runs `build` → waits for **manual approval** on the `pypi`
environment → publishes to PyPI → creates GitHub Release with
release notes from CHANGELOG.

### 5. Post-release verification

In a fresh venv:

```bash
pip install abides-ng==2.6.0
abides --help
```

Verify the PyPI project page renders the README correctly:
<https://pypi.org/project/abides-ng/2.6.0/>.

## Hotfix releases

For an urgent fix on a released line:

1. Branch from the release tag: `git checkout -b hotfix/2.6.1 v2.6.0`.
2. Apply the fix, update CHANGELOG and both `pyproject.toml`
   versions to `2.6.1`.
3. Open a PR back into `main`.
4. After merge, tag `v2.6.1` and push.

## Yanking a release

If a published version is broken, **yank** rather than delete (PyPI
does not allow re-uploading the same version):

1. Go to <https://pypi.org/manage/project/abides-ng/releases/>
2. Find the affected version → "Options" → "Yank release".
3. Provide a brief reason. Yanked versions remain installable by
   exact pin but are excluded from `pip install abides-ng`
   resolution.
4. Cut a fixed `2.6.x` patch as soon as possible.

## What is intentionally not in the workflow yet

The following are deferred to follow-up work:

- **Sigstore attestations** for wheel provenance.
- **`pip-audit` and Dependabot** for supply-chain scanning.
- **Conda-forge publication.**
- **Sphinx docs build and deploy** (currently docs live in `docs/`
  and on GitHub).
