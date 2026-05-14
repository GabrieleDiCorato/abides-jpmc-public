# Rebrand to abides-ng

## Decision summary

| Item | Before | After |
|---|---|---|
| Primary PyPI wheel | `abides-markets` (was planned) | **`abides-ng`** |
| Gym PyPI wheel | deferred | **`abides-ng[gym]`** optional extra (still deferred) |
| Import names | `abides_core`, `abides_markets`, `abides_gym` | **unchanged** |
| Repo name | `abides-hasufel` | **`abides-ng`** (manual GitHub rename, user-owned) |
| Source directories | `abides-markets/`, `abides-gym/` | **unchanged** (dirs are internal; wheel name ≠ dir name) |

### Wheel design rationale

`pip install abides-ng` (two tokens, clean). When abides-gym is validated
and ready to ship, it becomes `pip install abides-ng[gym]` — one PyPI project,
two profiles. The `[gym]` optional extra declares `gymnasium` and `ray[rllib]`
as dependencies. The `abides_gym` source will be force-included into the wheel
at that point. No three-token names anywhere.

---

## Files to change

### 1. `abides-markets/pyproject.toml`
- `name = "abides-ng"` (was "abides-markets")
- `description`: update to drop "kernel and market microstructure" framing → project-level
- Add `[project.optional-dependencies]` with `gym` extra as a forward-declaration
  (deps listed, source not yet bundled — gym remains deferred)
- Update all `project.urls` from `.../abides-markets/` → `.../abides-ng/`
- `[project.scripts]`: no change (`abides = "abides_core.abides:main"`)
- `[tool.hatch.build.targets.wheel]`: no change (force-include and packages unchanged)

### 2. `abides-gym/pyproject.toml`
- `dependencies`: `"abides-markets==2.6.0"` → `"abides-ng==2.6.0"`
- (Not published; kept only as a workspace dev target)

### 3. Root `pyproject.toml`
- `[tool.uv.sources]`: `abides-markets` key → `abides-ng`
- `[dependency-groups].dev`: `"abides-markets[test,docs]"` → `"abides-ng[test,docs]"`

### 4. `.github/workflows/release.yml`
- Version-check step: update variable name `MARKETS_VER` → `NG_VER` and
  grep from `abides-markets/pyproject.toml` (path unchanged)
- `environment.url` in both publish jobs: `.../abides-markets/` → `.../abides-ng/`

### 5. `README.md`
- Install section: `pip install abides-ng` everywhere
- About this Fork section: update PyPI reference
- Any other `abides-markets` references in install context

### 6. `CHANGELOG.md` — v2.6.0 Distribution Rename section
- Rewrite to reflect `abides-ng` as the wheel name
- Explain the two-token choice and the optional-extras plan for gym

### 7. `docs/project/release-process.md`
- All references to `abides-markets` wheel → `abides-ng`
- PyPI setup section: project name → `abides-ng`
- Local verification section: update twine check, pip install examples

### 8. `abides-markets/README.md`
- Update install instructions

### 9. `.github/copilot-instructions.md` (workspace instructions)
- Update any distribution name references if present

---

## What does NOT change

- Source directory names (`abides-markets/`, `abides-gym/`, `abides-core/`)
- Import package names (`abides_core`, `abides_markets`, `abides_gym`)
- Internal module structure
- Test suite
- Version (`2.6.0`)
- Gym deferral status

---

## Gym optional-extra design (forward declaration)

In `abides-ng`'s pyproject.toml, declare the extras now so the metadata
is correct when gym ships:

```toml
[project.optional-dependencies]
gym = [
    "gymnasium>=1.0.0",
    "ray[rllib]>=2.40.0",
]
```

When gym is validated:
1. Add `force-include` for `../abides-gym/abides_gym` → `abides_gym` in the wheel
2. Remove `abides-gym` from workspace members (or keep for dev convenience)
3. Ship `abides-ng` patch release

---

## Step list

1. Edit `abides-markets/pyproject.toml` — rename, add gym extra
2. Edit `abides-gym/pyproject.toml` — update dep name
3. Edit root `pyproject.toml` — uv.sources + dependency-groups
4. Edit `.github/workflows/release.yml` — URLs + variable names
5. Edit `README.md` — install instructions
6. Edit `CHANGELOG.md` — v2.6.0 Distribution Rename section
7. Edit `docs/project/release-process.md` — all wheel name references
8. Edit `abides-markets/README.md` — install instructions
9. Run `uv sync` to update lockfile
10. Run `uv build --wheel` from `abides-markets/` → verify wheel name in `dist/`
11. Run `uv run twine check dist/*`
12. Pre-commit clean → commit
