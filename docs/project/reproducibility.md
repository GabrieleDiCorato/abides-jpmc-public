# Reproducibility Policy

ABIDES is a discrete-event simulation. Reproducibility — the
property that two runs with the same configuration and the same
master seed produce bit-for-bit identical event streams, log files,
and metric outputs — is a **first-class guarantee**.

Industry users (research, model-validation, regulatory backtesting)
depend on this property to compare strategies fairly, to bisect
performance regressions, and to satisfy audit requirements.

## What is guaranteed

Within a given release line (`X.Y.z` for any patch `z`):

1. **Identical seeds → identical runs.** Given the same
   `SimulationConfig` (or compiled `runtime`), the same master
   seed, the same Python version, and the same `numpy` version,
   two runs produce byte-identical:
   - Event sequences and timestamps
   - Log writer outputs (BZ2 pickle contents)
   - Custom-state dictionaries
   - L1/L2 book histories
   - Agent return values
2. **Hierarchical RNG derivation.** Per-component RNGs are derived
   deterministically from the master seed via
   `_derive_seed(master_seed, component_label)`. Adding a new
   stochastic component must use a new label so existing
   components' seeds are unaffected.
3. **No implicit RNG draws.** The kernel and infrastructure code
   never consume RNG state outside of explicit, labelled draws.
   Latency-noise draws are gated on `noise is not None` (see
   v2.6.0 changelog for the historical fix).

## What is NOT guaranteed

- **Across major versions.** v3.x runs may diverge from v2.x even
  with the same seed and config. Major bumps are explicitly the
  vehicle for accumulated reproducibility-affecting changes.
- **Across minor versions** when the changelog explicitly notes a
  *Breaking Change* in the `Reproducibility` category. Such
  changes should be **rare** and exhaustively documented.
- **Across `numpy` major versions.** RNG implementations in numpy
  itself can change. Pin `numpy<3` (already done in our wheel
  metadata) to stay on the same line.
- **Across Python versions** if any code path depends on dict
  iteration order in a way the kernel does not control. We aim
  for Python 3.11+ compatibility but do not promise byte-for-byte
  parity across Python minors.
- **Walltime measurements.** Latency is in simulation time, not
  walltime; walltime varies by hardware. Performance metrics
  derived from walltime (e.g. `events_per_second`) are
  intentionally non-reproducible.

## Process for changes that affect reproducibility

A change "affects reproducibility" if it could alter the output of
any seeded simulation. This includes:

- Any change to RNG draw order or label.
- Any change to event ordering or message-priority semantics.
- Any change to floating-point computation order in hot paths
  (typically harmless for integer-cents prices, but watch for
  `numpy` ufunc ordering).
- Any change to default parameter values for stochastic components.
- Any change to oracle behaviour or to agent wakeup scheduling.

### Required steps

1. **Discuss in an issue first.** Reproducibility-affecting changes
   need explicit acknowledgement before implementation.
2. **Major-version bump required.** A new major version (or an
   announced reproducibility-break window during pre-release) is
   the only path. Do not slip such a change into a minor or patch
   release.
3. **Migration documentation in CHANGELOG.** The
   `Breaking Changes` section must include:
   - A description of what changed.
   - The reason the change is necessary.
   - A concrete migration recipe (if a flag or constructor kwarg
     can restore the legacy behaviour, document it).
   - A note on which tests were updated and why.
4. **Update reproducibility tests.** Tests like
   `abides-core/tests/test_seed_replicability.py` compare two
   runs to each other (not to a hardcoded baseline), so they
   typically continue to pass. If you add a new reproducibility
   test for a specific scenario, follow the same pattern: compare
   two runs, not a hardcoded snapshot.

## Reporting reproducibility regressions

If you observe two runs with identical seeds + config diverging
on the same release line and the same Python / numpy versions,
**this is a bug**. Report it via the standard issue tracker with:

- The configuration (or a minimal reproducer).
- The master seed.
- The Python and `numpy` versions.
- The diff between the two run outputs (event sequence,
  custom-state, or log output).

We treat unintentional reproducibility regressions as bugs that
warrant a patch release.

## Historical context

* **v2.6.0** removed the no-op default latency-noise RNG draw —
  the first explicitly-acknowledged reproducibility break, retained
  behind an opt-in flag (`noise=[1.0]` on
  `UniformLatencyModel`/`MatrixLatencyModel`).
* Earlier releases (pre-v2.6.0) did not have a documented
  reproducibility policy; runs are expected to be reproducible on
  the same release but no formal guarantee was made.

This policy applies from v2.6.0 forward.
