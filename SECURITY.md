# Security Policy

## Reporting a vulnerability

If you discover a security vulnerability in ABIDES, **please do not
open a public issue.** Instead, report it privately via one of:

- **GitHub Security Advisories** (preferred):
  <https://github.com/GabrieleDiCorato/abides-ng/security/advisories/new>
- **Email**: gabriele.dicorato@gmail.com — please include
  `[abides-security]` in the subject line.

## What to include

- A description of the issue and its potential impact.
- Steps to reproduce (a minimal proof-of-concept is ideal).
- The version of `abides-ng` affected (e.g. `pip show abides-ng`).
- Your environment (Python version, OS) if relevant.

## Response cadence

This project is solo-maintained on a best-effort basis:

- **Acknowledgement**: typically within 7 days.
- **Triage and initial assessment**: within 14 days.
- **Fix or mitigation**: depends on severity and complexity; we'll
  keep you updated.

If you do not hear back within 14 days, please follow up via the
same channel.

## Disclosure policy

We follow **coordinated disclosure**:

1. Reporter notifies us privately.
2. We confirm the issue, develop a fix, and prepare a release.
3. Public disclosure (advisory + patched release) once the fix is
   available, ideally within 90 days of the initial report.

Reporters who wish to be credited will be acknowledged in the
GitHub Security Advisory and the `CHANGELOG.md` entry.

## Supported versions

| Version | Status                  |
|---------|-------------------------|
| 2.6.x   | Active — security fixes |
| < 2.6   | Unsupported             |

When v3.0 ships, the previous minor (`2.6.x`) will receive security
fixes only for a transitional period announced in the v3.0 release
notes.

## Scope

In scope:

- Vulnerabilities in the published `abides-ng` wheel.
- Insecure defaults that could affect downstream simulations
  (e.g. predictable RNG, log-injection paths).
- Supply-chain issues in our build/release pipeline.

Out of scope:

- Vulnerabilities in third-party dependencies — please report those
  upstream first; we'll bump dependency versions promptly once a
  fixed release is available.
- Issues that require local code execution, debugger access, or
  modification of the user's environment to exploit.
- Denial-of-service via maliciously crafted simulation configs
  (configs are explicitly trusted input).
