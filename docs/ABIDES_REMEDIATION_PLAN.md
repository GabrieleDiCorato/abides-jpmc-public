# ABIDES Performance Remediation Plan

## Overview
This plan addresses validated performance and architectural bottlenecks in ABIDES, prioritizing fixes by impact and effort. Each step includes rationale, estimated effort, and expected speedup.

---

## 1. Latency Matrix Aliasing
- **Problem:** Latency matrix is aliased, causing unintended cross-agent latency changes.
- **Remediation:** Deep copy latency matrix for each agent.
- **Effort:** Low
- **Expected Speedup:** Medium (prevents subtle bugs, improves simulation accuracy)

---

## 2. Message Batch Delay Bug
- **Problem:** Message batch delay logic is incorrect, causing event queue misordering.
- **Remediation:** Refactor batch delay handling in kernel and agent message processing.
- **Effort:** Medium
- **Expected Speedup:** High (reduces event queue churn, improves determinism)

---

## 3. Global Logging Config
- **Problem:** Global logging config mutates logger state, causing thread-safety issues.
- **Remediation:** Move to per-agent logger setup, avoid global mutation.
- **Effort:** Medium
- **Expected Speedup:** Medium (improves thread-safety, reduces log contention)

---

## 4. Subscription Scaling
- **Problem:** Subscription management scales poorly with agent count.
- **Remediation:** Refactor subscription loops to use sets/dicts, avoid O(N^2) scans.
- **Effort:** Medium
- **Expected Speedup:** High (improves scalability for large agent counts)

---

## 5. Order Book O(N) Insertion
- **Problem:** Order book insertion/cancellation is O(N), causing slowdowns with large books.
- **Remediation:** Use bisect or heap for order insertion, optimize cancellation.
- **Effort:** High
- **Expected Speedup:** High (critical for high-frequency simulation)

---

## 6. Book Logging Memory Churn
- **Problem:** Book logging creates excessive memory churn, slowing simulation.
- **Remediation:** Refactor book logging to minimize deep copies and redundant data.
- **Effort:** Medium
- **Expected Speedup:** Medium (reduces memory usage, improves speed)

---

## 7. Deep Copy Overhead
- **Problem:** Frequent deep copies in event queue and book logging.
- **Remediation:** Replace deep copy with shallow copy or custom copy logic where safe.
- **Effort:** Medium
- **Expected Speedup:** Medium (reduces CPU and memory overhead)

---

## 8. Filter Inefficiency
- **Problem:** Inefficient filtering in event queue and agent message handling.
- **Remediation:** Use generator expressions and optimized data structures.
- **Effort:** Low
- **Expected Speedup:** Medium (improves event processing speed)

---

## 9. Oracle RNG Policy
- **Problem:** Oracle random state policy is inconsistent, risking nondeterminism.
- **Remediation:** Enforce injected random state for all oracles and per-symbol RNGs.
- **Effort:** Low
- **Expected Speedup:** Low (improves reproducibility)

---

## Next Steps
1. Implement fixes in priority order.
2. Validate with profiling and test suites.
3. Document changes and update developer guide.

---

## Appendix
- [Conversation summary](docs/ABIDES_LLM_INTEGRATION_GOTCHAS.md)
- [Implementation guide](docs/ABIDES_CUSTOM_AGENT_IMPLEMENTATION_GUIDE.md)
- [Data extraction](docs/ABIDES_DATA_EXTRACTION.md)
