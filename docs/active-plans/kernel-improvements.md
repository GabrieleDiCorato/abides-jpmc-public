# Kernel Improvement Plan

**Target file:** `abides-core/abides_core/kernel.py`
**Status:** Ready to execute PR by PR.
**Audience:** An implementing agent that has *not* read the kernel in detail.

This document is the contract. Every PR section below is self-contained:
read it, evaluate it critically against the code, run the listed checks, open the PR.
Do **not** combine PRs. Do **not** invent scope. If something contradicts
reality, stop and ask before improvising.

---

## 0. Context for the Implementer (read this first)

### What `Kernel` is

ABIDES is a **discrete-event** market simulator. Time advances in
nanoseconds (`int`). There is no real-time loop and no thread of
execution per agent — instead, the `Kernel` owns:

- A list of `Agent` instances (`self.agents`), indexed by `agent.id`.
  By convention `agents[i].id == i`. Nothing currently validates this.
- A min-heap of pending events (`self.messages`), each entry
  `(deliver_at, (sender_id, recipient_id, message))`.
- Two parallel arrays sized `N = len(agents)`:
  - `agent_current_times[i]` — the next time agent *i* is free to
    process anything (it's "busy" until then).
  - `agent_computation_delays[i]` — how long agent *i* takes to
    process one event.
- A latency model — how long a message takes to travel from sender
  to recipient.

`runner()` is the hot loop:

```
while heap not empty and time < stop_time:
    deliver_at, (sender_id, recipient_id, message) = heappop(messages)
    if agent[recipient_id] busy past deliver_at: re-push at busy-until time
    else: deliver message → agent.wakeup() or agent.receive_message()
          advance agent_current_times[recipient_id]
```

Agents talk to each other only by enqueueing messages
(`kernel.send_message(...)`) and scheduling future self-wakeups
(`kernel.set_wakeup(...)`). Both go through the heap.

### Why this kernel is fragile today

1. **God class.** ~800 lines, eight responsibilities (event loop,
   lifecycle, agent registry, per-agent state, latency, logging,
   financial statistics, gym integration). Every responsibility is
   exposed as bare attributes that downstream code mutates.
2. **Parallel arrays with an unchecked invariant.** Five lists assume
   `agents[i].id == i`. A user passing `Kernel(agents=[a3, a1, a2])`
   gets silently wrong routing.
3. **Two real correctness bugs that have been in the code for years
   and would silently corrupt simulations under specific call patterns.**
4. **Encapsulation by convention.** Mutable defaultdicts, lists of
   dicts, `setattr(self, key, value)` from `custom_properties` —
   anyone can stomp on anything.

### How to verify things in this codebase

- **Find a symbol's call sites:** `grep -rn "<symbol>" abides-*/`
- **Run the kernel test suite:** `pytest abides-core/tests/ -x -q`
- **Run all unit tests:** `pytest abides-core/tests/ abides-markets/tests/ -x -q`
- **Smoke a config:** `python -m abides_core.abides --help`

### Simulation primitives you will touch

| Object | File | Role |
|---|---|---|
| `Kernel` | `abides-core/abides_core/kernel.py` | Event loop owner |
| `Agent` | `abides-core/abides_core/agent.py` | Base class for everything that participates |
| `Message` | `abides-core/abides_core/message.py` | Carrier for any inter-agent event |
| `WakeupMsg` | `abides-core/abides_core/message.py` | Marker class for self-wakeups |
| `MessageBatch` | `abides-core/abides_core/message.py` | Wrapper to deliver several messages atomically |
| `LatencyModel` | `abides-core/abides_core/latency_model.py` | Network delay model |
| `abides.run()` | `abides-core/abides_core/abides.py` | The single official entry point used by configs |
| `TradingAgent` | `abides-markets/abides_markets/agents/trading_agent.py` | Markets-specific agent base |

### Useful invariants you can rely on

- All current `Kernel(...)` constructions in the codebase already use
  keyword arguments. The keyword-only `__init__` change in PR 7 is mechanical.
- `Agent.delay()` has **zero call sites in the codebase today**. The dispatch
  ordering bug in PR 3 is latent; fixing it is safe.
- `Message.message_id` is read by no agent and no test asserts on it.
  It exists only as a heap tiebreaker. The refactor in PR 3 is safe.
- The oracle is injected via `custom_properties={"oracle": ...}` and
  read by `ExchangeAgent` and `ValueAgent` as `self.kernel.oracle`.
  Do not break this until the planned `extras` migration (out of scope
  for this plan).

---

---

## 1. PR Plan (execute in order)

Each PR has the same structure:

- **Goal**
- **Files touched** (exact paths)
- **Pre-flight checks** (commands to run before editing)
- **Steps** (mechanical edits)
- **Tests** (what to add)
- **Acceptance** (what must pass)
- **Risk notes**

> Run the pre-flight grep for every PR before editing. If results
> differ from what the plan claims, **stop and ask** — the codebase
> may have moved.

---

### PR 1 — Hygiene fixes (no behaviour change for valid configs)

**Goal:** Land all low-risk safety and correctness fixes in one PR.

**Files:** `abides-core/abides_core/kernel.py`, `abides-core/tests/test_kernel.py`.

**Pre-flight:**
```
grep -rn "kernel\.agents\b" abides-markets abides-gym | head
grep -rn "agent_latency:" abides-core abides-markets
```
(Confirm nobody iterates `kernel.agents` assuming a particular ordering.)

**Steps:**

1. **`write_summary_log` must honour `skip_log`.**
   Top of `write_summary_log`, add:
   ```python
   if self.skip_log:
       return
   ```

2. **Replace gym `assert` with `ValueError`.**
   In `__init__`, replace
   ```python
   assert (
       len(self.gym_agents) <= 1
   ), "ABIDES-gym currently only supports using one gym agent"
   ```
   with
   ```python
   if len(self.gym_agents) > 1:
       raise ValueError(
           "ABIDES-gym currently supports at most one gym agent; "
           f"got {len(self.gym_agents)}"
       )
   ```
   (Asserts are stripped under `python -O`; this is config validation.)

3. **Drop dead assertion in the hot loop.**
   Search `runner()` for `assert self.current_time is not None` and
   delete it. The variable is set immediately above.

4. **Declare every attribute in `__init__`.**
   Audit `runner()`, `initialize()`, `terminate()` for attributes
   first-assigned outside `__init__`. Known offenders:
   - `event_queue_wall_clock_start` (set in `initialize()`)
   - `ttl_messages` (set in `initialize()`)
   Add to `__init__` with `None`/`0` defaults and proper type hints.

5. **`ZeroDivisionError` guard in `terminate()`.**
   In the mean-reporting block:
   ```python
   for k, v in self.mean_result_by_agent_type.items():
       count = self.agent_count_by_type.get(k, 0)
       if count == 0:
           continue
       logger.info(f"Mean ending value by agent type: {k} : {v / count}")
   ```
   PR 4 will rewrite this whole block. PR 1 just stops it crashing on
   an empty simulation.

6. **Validate the `agent.id == index` invariant.**
   Top of `__init__`, immediately after the `agents` parameter is
   accepted:
   ```python
   for idx, agent in enumerate(agents):
       if agent.id != idx:
           raise ValueError(
               f"agents[{idx}].id == {agent.id}, expected {idx}. "
               f"Kernel requires agents to be ordered by id, with ids 0..N-1."
           )
   ```

7. **Type annotation and docstring fixes.**
   - `default_latency: float = 1` → `default_latency: int = 1`
   - `agent_latency: list[list[float]]` → `list[list[int]]`
   - In `runner()` docstring: `"results"` → `"result"` (singular).
   - Add a docstring line on `__init__` documenting the
     `agent.id == index` invariant.

**Tests (add to `abides-core/tests/test_kernel.py`):**

- `test_write_summary_log_respects_skip_log` — construct with
  `skip_log=True`, call `write_summary_log()`, assert no file is
  created in CWD.
- `test_too_many_gym_agents_raises_value_error` — pass two fake
  agents whose direct base name is `"CoreGymAgent"`, expect
  `ValueError` (not `AssertionError`).
- `test_terminate_zero_count_does_not_crash` — empty kernel, call
  `terminate()`, no exception.
- `test_agent_id_invariant_violation_raises` — pass two agents with
  swapped ids, expect `ValueError`.

**Acceptance:**
- `pytest abides-core/tests/ abides-markets/tests/ -x -q` green.
- `python -O -m pytest abides-core/tests/test_kernel.py -x -q` green
  (proves the gym check works under `-O`).

**Risk:** Minimal. None of these change behaviour for a correctly-built
simulation.

---

### PR 2 — State validation & reset hygiene

**Goal:** Make kernel state predictable across resets and reject
invalid `custom_properties`.

**Files:** `abides-core/abides_core/kernel.py`, `abides-core/abides_core/abides.py`, `abides-core/tests/test_kernel.py`.

**Pre-flight:**
```
grep -rn "custom_properties\s*=" abides-* notebooks
grep -rn "kernel\.reset\(" abides-*
grep -rn "Kernel(" abides-* notebooks | grep -v "random_state\|seed"
```
The third grep finds Kernel constructions without explicit seeding
(notebooks are the most likely offenders). Note them — they will
get a deprecation warning, not a hard error.

**Steps:**

1. **Block reserved keys in `custom_properties`.**
   Define at module level in `kernel.py`:
   ```python
   _KERNEL_RESERVED_ATTRS: frozenset[str] = frozenset({
       "agents", "messages", "current_time", "start_time", "stop_time",
       "kernel_wall_clock_start", "event_queue_wall_clock_start",
       "summary_log", "random_state", "skip_log", "log_dir",
       "ttl_messages", "has_run", "custom_state",
       "mean_result_by_agent_type", "agent_count_by_type",
       "gym_agents", "agent_latency", "latency_noise",
       "agent_latency_model", "agent_current_times",
       "agent_computation_delays", "current_agent_additional_delay",
   })
   ```
   In `__init__`, after `custom_properties = custom_properties or {}`:
   ```python
   bad = set(custom_properties).intersection(_KERNEL_RESERVED_ATTRS)
   if bad:
       raise ValueError(
           f"custom_properties keys collide with kernel internals: "
           f"{sorted(bad)}. Reserved names: {sorted(_KERNEL_RESERVED_ATTRS)}"
       )
   ```
   **Critical:** `oracle` is NOT in the blocklist. Verify
   `kernel.oracle` injection still works by running
   `pytest abides-markets/tests/test_config_system.py -x -q`.

2. **`initialize()` clears simulation state.**
   Per decision (review round 2): clearing lives in `initialize()`, not
   `reset()`. `reset()` already calls `terminate()` then `initialize()`,
   so any future re-initialization path benefits from the same clean
   slate. At the top of `initialize()`, before the
   `kernel_initializing` loop:
   ```python
   self.agent_current_times[:] = [self.start_time] * len(self.agents)  # list form; slice assign after PR 6
   self.ttl_messages = 0
   self.custom_state.clear()
   self.summary_log.clear()
   self.messages.clear()
   self.current_agent_additional_delay = 0
   self._next_seq = 0   # only after PR 3 lands; remove this line if running PR 2 in isolation
   ```
   Note: `agent_computation_delays` is **not** cleared — it carries
   per-agent overrides from the constructor that must persist across
   `reset()`. If this turns out to be wrong (i.e., agents mutate it at
   runtime and want a fresh slate per episode), revisit in a follow-up.

   **Doc update (this PR):** add a paragraph to
   `docs/reference/llm-gotchas.md` documenting that `Kernel.initialize()`
   is the canonical reset point and what state it clears.

3. **Require explicit seed or `random_state`.**
   Replace the existing default-construction block in `__init__`:
   ```python
   if random_state is None:
       if seed is None:
           import warnings
           warnings.warn(
               "Kernel constructed without seed or random_state; using "
               "process-global numpy state. This will become an error "
               "in a future release. Pass seed=... or random_state=...",
               DeprecationWarning, stacklevel=2,
           )
           seed = int(np.random.randint(low=0, high=2**32, dtype="uint64"))
       random_state = np.random.RandomState(seed=seed)
   self.random_state = random_state
   ```
   Verify `abides.run()` is unaffected — it already passes
   `random_state=kernel_random_state or np.random.RandomState(seed=kernel_seed)`
   at `abides-core/abides_core/abides.py` L65.

   **CLI:** Leave the hardcoded `seed=1` alone in this PR. It will not warn.

**Tests:**

- `test_custom_properties_reserved_key_raises` — try
  `custom_properties={"agents": [...]}`, expect `ValueError`.
- `test_custom_properties_user_key_allowed` — pass
  `custom_properties={"oracle": object(), "my_thing": 42}`, expect
  no error and both attrs present on kernel.
- `test_reset_clears_state` — run a tiny sim, mutate
  `custom_state["x"] = 1`, call `reset()`, assert empty.
- `test_random_state_warns_when_unspecified` — construct with
  neither `seed` nor `random_state`, expect `DeprecationWarning`.

**Acceptance:**
- All tests green including
  `pytest abides-markets/tests/test_config_system.py -x -q`
  (oracle injection unaffected).

**Risk:** Low-medium. The deprecation warning may surface in notebooks.

---

### PR 3 — Fix dispatch ordering bug + heap refactor

**Goal:** Fix a silent correctness bug in the event dispatcher, and
replace the global message-counter with a per-kernel sequence number
to make simulations reproducible across resets and parallel runs.

**Background on the two bugs being fixed:**

*Dispatch ordering bug:* In `runner()`, the wakeup branch advances
`agent_current_times` *after* calling `wakeup()`, which is correct —
any `Agent.delay()` call inside `wakeup()` then shifts the agent's next
available time correctly. The non-wakeup branch does the opposite: it
advances `agent_current_times` *before* calling `receive_message()`, so
any `Agent.delay()` set inside a message handler is silently dropped.
No existing agent exercises this path today, making the bug latent but
a genuine footgun.

*Global counter bug:* `Message._message_id_generator` is a class-level
`itertools.count(1)`. Every `Message` ever constructed in the process —
across kernels, gym episodes, parallel tests — increments the same
counter. The counter is used as a heap tiebreaker, which means heap
ordering is coupled to process-global state. Two identical simulations
run back-to-back get different tiebreaker sequences; `kernel.reset()`
does not reset it.

**Files:**
- `abides-core/abides_core/kernel.py`
- `abides-core/abides_core/message.py`
- `abides-core/tests/test_kernel.py`

**Pre-flight:**
```
grep -rn "Message\.__lt__\|message_id" abides-* notebooks
grep -rn "self\.delay\(" abides-* notebooks
grep -rn "WakeupMsg(" abides-*
```
Confirm: `message_id` has only internal readers in `kernel.py`/`message.py`;
`self.delay()` is unused; `WakeupMsg()` is constructed only in `set_wakeup`.

**Steps:**

1. **Move heap ordering out of `Message` (`message.py`).**

   Replace the `_message_id_generator` class attribute and `__lt__` with
   a deprecation property:
   ```python
   class Message:
       def __init__(self) -> None:
           pass  # no more counter

       @property
       def message_id(self) -> int:
           import warnings
           warnings.warn(
               "Message.message_id is deprecated and will be removed; "
               "kernel ordering no longer relies on it.",
               DeprecationWarning, stacklevel=2,
           )
           return id(self)

       # __lt__ removed — kernel uses _HeapEntry for ordering.
   ```
   Delete the global `itertools.count` import if unused.

2. **Add `_HeapEntry` and per-kernel sequence counter (kernel side).**

   In `kernel.py`, top of module:
   ```python
   from dataclasses import dataclass, field

   @dataclass(order=True)
   class _HeapEntry:
       deliver_at: int
       seq: int
       sender_id: int = field(compare=False)
       recipient_id: int = field(compare=False)
       message: "Message" = field(compare=False)
   ```
   In `Kernel.__init__`:
   ```python
   self._next_seq: int = 0
   self.messages: list[_HeapEntry] = []
   ```
   In `reset()`:
   ```python
   self._next_seq = 0
   ```
   Add a private helper:
   ```python
   def _enqueue(self, deliver_at, sender_id, recipient_id, message) -> None:
       heapq.heappush(self.messages, _HeapEntry(
           deliver_at, self._next_seq, sender_id, recipient_id, message,
       ))
       self._next_seq += 1
   ```
   Replace every `heapq.heappush(self.messages, (deliver_at, (sender_id, recipient_id, message)))`
   in `send_message`, `set_wakeup`, and the in-future requeue path with
   `self._enqueue(...)`.

3. **`WakeupMsg` singleton.**
   Module-level:
   ```python
   _WAKEUP_SINGLETON = WakeupMsg()
   ```
   In `set_wakeup`, replace `WakeupMsg()` with `_WAKEUP_SINGLETON`.
   Wakeup messages carry no data; sharing one instance is safe and
   reduces per-wakeup allocation.

4. **Unified dispatch helper (fixes the dispatch ordering bug).**
   Replace the two branches of `runner()` with a call to `_dispatch()`:
   ```python
   def _dispatch(self, deliver_at, sender_id, recipient_id, message):
       rid = recipient_id
       # If recipient still busy, requeue to its next free moment.
       if self.agent_current_times[rid] > deliver_at:
           self._enqueue(self.agent_current_times[rid], sender_id, rid, message)
           return
       self.agent_current_times[rid] = deliver_at
       self.current_agent_additional_delay = 0
       if message.__class__ is WakeupMsg:
           self.agents[rid].wakeup(deliver_at)
       elif message.__class__ is MessageBatch:
           for sub in message.messages:
               self.agents[rid].receive_message(deliver_at, sender_id, sub)
       else:
           self.agents[rid].receive_message(deliver_at, sender_id, message)
       # Advance AFTER delivery so any Agent.delay() call inside
       # wakeup() / receive_message() takes effect on the agent's own slot.
       self.agent_current_times[rid] += (
           self.agent_computation_delays[rid]
           + self.current_agent_additional_delay
       )
   ```
   The `runner()` loop body becomes:
   ```python
   entry = heapq.heappop(self.messages)
   self.current_time = entry.deliver_at
   if self.current_time > self.stop_time:
       break
   self._dispatch(entry.deliver_at, entry.sender_id, entry.recipient_id, entry.message)
   self.ttl_messages += 1
   ```
   Note: `__class__ is X` is faster than `isinstance` and used here
   intentionally — do not "improve" to `isinstance`.

5. **Lazy debug logging.**
   Replace `logger.debug(f"...")` patterns inside the hot loop with
   `logger.debug("...", arg1, arg2)` style. Wrap any call that builds
   non-trivial debug strings in `if self.show_trace_messages:`.

6. **Periodic stats line: use `%s`-style formatting.**
   Find the periodic stats log line in `runner()` (prints messages/sec etc.)
   and switch to `%s`-style so format string evaluation only runs when
   the line is actually emitted at the configured level.

**Tests:**

- `test_delay_in_receive_message_advances_agent_time`:
  Build two agents A, B. B sends a message to A at t=100. A's
  `receive_message` calls `self.delay(500)`. Assert
  `kernel._agent_current_times[A.id] >= 100 + comp_delay + 500`.
- `test_sequence_resets_across_kernel_reinitialization`:
  Run kernel, capture the `_next_seq` value at end. Call `reset()`,
  re-run identical config. Assert `_next_seq` after second run equals
  the value after the first run (i.e. sequence is deterministic).
- `test_wakeup_singleton_reused`:
  `set_wakeup` twice, pop both heap entries, assert
  `entry1.message is entry2.message`.
- `test_message_batch_delivered_as_individual_messages`:
  Send one `MessageBatch` of 3 sub-messages. Assert recipient's
  `receive_message` is called 3 times with the sub-messages directly.

**Acceptance:**
- All related tests green.

**Risk:** Medium. Touches the hot loop. Run the full
`pytest abides-markets/tests/ -x -q` suite. If anything in
`test_market_boundaries.py` breaks, it is likely the `FakeKernel` mock —
adapt by giving it a `_enqueue` method or a `messages` list (not by
reverting the dispatch refactor).

---

### PR 4 — Move financial metrics out of core

**Goal:** The kernel currently carries two financial-statistics
attributes (`mean_result_by_agent_type`, `agent_count_by_type`) that
have nothing to do with event routing. Move this responsibility to a
generic `Agent.report_metric()` method that stores results in
`kernel.custom_state`, and remove the leaky attributes from the kernel.

**Files:**
- `abides-core/abides_core/kernel.py`
- `abides-core/abides_core/agent.py`
- `abides-markets/abides_markets/agents/trading_agent.py`
- `abides-markets/tests/test_market_boundaries.py`

**Pre-flight:**
```
grep -rn "mean_result_by_agent_type\|agent_count_by_type" abides-* notebooks
```

**Steps:**

1. **Add `Agent.report_metric()` in `agent.py`.**
   ```python
   def report_metric(self, key: str, value: float) -> None:
       """Aggregate a numeric metric by (agent_type, key) on the kernel.

       Stored in self.kernel.custom_state["agent_type_metrics"][type][key]
       as {"sum": float, "count": int}.
       """
       stats = self.kernel.custom_state.setdefault("agent_type_metrics", {})
       bucket = stats.setdefault(self.type, {})
       cell = bucket.setdefault(key, {"sum": 0.0, "count": 0})
       cell["sum"] += float(value)
       cell["count"] += 1
   ```

2. **Remove the two legacy kernel attributes.**
   In `kernel.py.__init__`, delete:
   ```python
   self.mean_result_by_agent_type: defaultdict[str, int] = defaultdict(int)
   self.agent_count_by_type: defaultdict[str, int] = defaultdict(int)
   ```
   In `terminate()`, replace the existing mean-printing block with:
   ```python
   metrics = self.custom_state.get("agent_type_metrics", {})
   for agent_type, by_key in metrics.items():
       for key, cell in by_key.items():
           count = cell["count"]
           if count == 0:
               continue
           logger.info(
               "Mean %s by agent type %s: %s",
               key, agent_type, cell["sum"] / count,
           )
   ```
   Drop the `defaultdict` import if no longer used.

3. **Update `TradingAgent`.**
   In `abides-markets/abides_markets/agents/trading_agent.py`,
   verify the line numbers, then replace:
   ```python
   self.kernel.mean_result_by_agent_type[self.type] += gain
   self.kernel.agent_count_by_type[self.type] += 1
   ```
   with:
   ```python
   self.report_metric("ending_value", gain)
   ```

4. **Simplify `FakeKernel`** in
   `abides-markets/tests/test_market_boundaries.py`.
   Remove the two `defaultdict` attrs. Keep `custom_state = {}`.

**Tests:**

- `test_report_metric_aggregates_by_type_and_key` — two agents of
  same type, each calls `report_metric("x", 5)`. Assert
  `custom_state["agent_type_metrics"][type]["x"] == {"sum": 10.0, "count": 2}`.
- `test_terminate_prints_mean_metrics_no_legacy_attrs` — verify
  `kernel` has no `mean_result_by_agent_type` attribute and the
  mean log line is still emitted.

**Acceptance:**
- `pytest abides-markets/tests/test_market_boundaries.py -x -q` green.
- `pytest abides-markets/tests/ -x -q` green.

**Risk:** Medium. Touches markets layer. Notebooks that read
`kernel.mean_result_by_agent_type` directly will break — add a
DeprecationWarning shim if grep finds any (none expected).

---

### PR 5a — Latency model unification, behavior-preserving

**Goal:** Always go through a `LatencyModel`. Remove the legacy matrix
duplication. **Preserve current RNG-consumption behavior** so seeded
simulations remain bit-identical to pre-PR runs.

**Reproducibility constraint:** Today, `send_message` calls
`self.random_state.choice(len(self.latency_noise), p=self.latency_noise)`
on every send, *including the default `[1.0]` no-noise case*. This
consumes one RNG draw per message. PR 5a must keep this draw
(equivalently: route it through the latency model) so that
`test_seed_replicability.py` and any externally-seeded users do not
diverge.

**Files:**
- `abides-core/abides_core/latency_model.py`
- `abides-core/abides_core/kernel.py`
- `abides-core/tests/test_kernel.py`
- (Audit) `abides-markets/abides_markets/` for callers building matrix
  latency.

**Pre-flight:**
```
grep -rn "agent_latency\b\|latency_noise\b" abides-* notebooks
grep -rn "LatencyModel\|get_latency" abides-* notebooks
```

**Steps:**

1. **Add `UniformLatencyModel` and `MatrixLatencyModel`** in
   `latency_model.py`. Both accept an optional `noise` list and a
   `random_state` reference, and always consume one draw from it
   (preserving the existing RNG behavior):
   ```python
   class UniformLatencyModel(LatencyModel):
       def __init__(self, latency: int) -> None:
           self._latency = int(latency)
       def get_latency(self, sender_id: int, recipient_id: int) -> int:
           return self._latency

   class MatrixLatencyModel(LatencyModel):
       def __init__(self, matrix: np.ndarray) -> None:
           self._m = np.ascontiguousarray(matrix, dtype=np.int64)
       def get_latency(self, sender_id: int, recipient_id: int) -> int:
           return int(self._m[sender_id, recipient_id])
   ```
   Change the base-class return type annotation to `int`. Update the
   cubic model accordingly (cast to int at the boundary).

2. **Normalize legacy params into a model in `Kernel.__init__`.**
   ```python
   if agent_latency_model is None:
       if agent_latency is not None:
           agent_latency_model = MatrixLatencyModel(np.asarray(agent_latency))
       else:
           agent_latency_model = UniformLatencyModel(default_latency)
   self.agent_latency_model = agent_latency_model
   ```
   Drop the `agent_latency` and `latency_noise` attributes (or keep as
   `None` deprecated shims for one release).

3. **Preserve the noise RNG draw inside the model.**
   Both `UniformLatencyModel` and `MatrixLatencyModel` accept a
   `noise: list[float] | None` constructor arg and a `random_state`
   reference (passed by the kernel at construction or via `attach`):
   ```python
   class UniformLatencyModel(LatencyModel):
       def __init__(self, latency: int, noise: list[float] | None = None) -> None:
           self._latency = int(latency)
           self._noise = noise if noise is not None else [1.0]
       def get_latency(self, sender_id, recipient_id, *, random_state) -> int:
           draw = int(random_state.choice(len(self._noise), p=self._noise))
           return self._latency + draw
   ```
   The default `[1.0]` noise list **stays** in PR 5a — its removal is
   PR 5b. Update `LatencyModel.get_latency` signature to accept
   `random_state` (keyword-only) so the kernel can pass
   `self.random_state` on every call.

3. **Send path simplification.**
   ```python
   latency = self.agent_latency_model.get_latency(
       sender_id, recipient_id, random_state=self.random_state,
   )
   deliver_at = sent_time + latency
   self._enqueue(deliver_at, sender_id, recipient_id, message)
   ```

**Tests:**

- `test_uniform_latency_model_returns_constant_with_default_noise`
  (asserts the `[1.0]` draw is consumed: same RNG seeded twice gives
  identical sequences with N sends).
- `test_matrix_latency_model_returns_pairwise`
- `test_kernel_default_uses_uniform_latency_model`
- `test_legacy_agent_latency_param_wraps_into_matrix_model`
- `test_seed_replicability.py` **must pass unmodified** — this is the
  acceptance gate for PR 5a.

**Acceptance:**
- All existing latency tests pass unmodified.
- `pytest abides-markets/tests/test_seed_replicability.py -x -q` green
  with no baseline regeneration.
- `pytest abides-markets/tests/ -x -q` green.

**Doc update (this PR):** in `docs/reference/llm-gotchas.md`, document
that the kernel passes `random_state` to `LatencyModel.get_latency` and
that custom latency models must accept it as a keyword arg.

**Risk:** Medium. Custom `LatencyModel` subclasses outside the repo
will need to add `random_state` to their `get_latency` signature.

---

### PR 5b — Remove the no-op latency-noise RNG draw (reproducibility break)

**Goal:** Eliminate the `[1.0]` default noise allocation and its RNG
draw. **This is an explicit, documented reproducibility break** —
seeded simulations will produce different outputs after this PR than
before. Ship as a separate PR so the diff is reviewable as a single
intent.

**Files:**
- `abides-core/abides_core/latency_model.py`
- `abides-markets/tests/test_seed_replicability.py` (regenerate baselines)
- `CHANGELOG.md` (BREAKING entry)

**Steps:**

1. Change the `noise` default on `UniformLatencyModel` and
   `MatrixLatencyModel` from `[1.0]` to `None`.
2. In `get_latency`, when `self._noise is None`, return `self._latency`
   directly — no RNG call.
3. Regenerate `test_seed_replicability.py` golden values. Inspect the
   diff in the test file; the new values become the baseline.
4. Add a `## [Unreleased]` BREAKING entry in `CHANGELOG.md` describing
   the divergence and pointing users at how to restore the old
   behavior (pass `noise=[1.0]` explicitly).

**Acceptance:**
- `pytest abides-markets/tests/ -x -q` green with regenerated baselines.
- CHANGELOG entry present.
- One-paragraph rationale in `docs/reference/llm-gotchas.md` under a
  "reproducibility" section.

**Risk:** High symbolically (breaks reproducibility) but low
mechanically (one default value change + baseline refresh). Must NOT
be combined with any other PR.

---

### PR 6 — Convert per-agent state to numpy arrays

**Goal:** Make `agent_current_times` and `agent_computation_delays`
numpy arrays for fast access; index agents by type for O(1) lookup.

**Files:**
- `abides-core/abides_core/kernel.py`
- `abides-core/tests/test_kernel.py`

**Pre-flight:**
```
grep -rn "agent_current_times\|agent_computation_delays" abides-* notebooks
grep -rn "find_agents_by_type" abides-* notebooks
```
Expected: only `kernel.py` writes these arrays; only
`test_kernel.py` (L95, L121) reads them externally;
`find_agents_by_type` has one caller (`trading_agent.py` L237).

**Steps:**

1. **Convert to numpy with deprecation shims.**
   In `__init__`:
   ```python
   n = len(agents)
   self._agent_current_times: np.ndarray = np.full(n, self.start_time, dtype=np.int64)
   self._agent_computation_delays: np.ndarray = np.full(n, default_computation_delay, dtype=np.int64)
   if per_agent_computation_delays:
       for aid, d in per_agent_computation_delays.items():
           self._agent_computation_delays[aid] = d
   ```
   Add read-only properties that warn once on access:
   ```python
   @property
   def agent_current_times(self) -> np.ndarray:
       _warn_deprecated_attr("agent_current_times")
       return self._agent_current_times.view()  # read-only view via flags

   @property
   def agent_computation_delays(self) -> np.ndarray:
       _warn_deprecated_attr("agent_computation_delays")
       return self._agent_computation_delays.view()
   ```
   Mark the views read-only:
   ```python
   def _readonly(arr: np.ndarray) -> np.ndarray:
       v = arr.view()
       v.flags.writeable = False
       return v
   ```

2. **Type index for `find_agents_by_type`.**
   ```python
   self._agents_by_type: dict[type, list[int]] = defaultdict(list)
   for idx, a in enumerate(agents):
       for cls in type(a).__mro__:
           self._agents_by_type[cls].append(idx)

   def find_agents_by_type(self, agent_type: type[Agent]) -> list[int]:
       return list(self._agents_by_type.get(agent_type, ()))
   ```

3. **Update internal references** in `kernel.py` to use `_agent_current_times`
   / `_agent_computation_delays` (not the property shims, which would
   trigger the warning on every dispatch).

4. **Update `test_kernel.py` L95, L121** to read
   `kernel._agent_current_times[1]` (the new internal name).

**Tests:**

- `test_per_agent_state_is_numpy_int64`
- `test_legacy_agent_current_times_attr_warns_and_is_readonly`
- `test_find_agents_by_type_uses_index_o1` (call N times against
  N=1000 agents, assert wall time is sub-millisecond)

**Acceptance:**
- All tests green.
- Memory: a 500-agent kernel uses two numpy arrays of 4 KB each
  instead of two Python lists of ~14 KB each (rough; document in PR).

**Risk:** Medium. Any external code that *wrote* to these lists will
silently fail (numpy view is read-only). Pre-flight grep should
confirm no writers exist.

---

### PR 7 — Architecture: LogWriter, lifecycle state machine, GymAdapter, keyword-only init

**Goal:** Decompose remaining responsibilities. Solidify the public
contract.

**Files (new):**
- `abides-core/abides_core/log_writer.py`
- `abides-core/abides_core/lifecycle.py`
- `abides-core/abides_core/gym_adapter.py`

**Files (modified):**
- `abides-core/abides_core/kernel.py`
- `abides-core/abides_core/abides.py`
- `abides-gym/abides_gym/envs/core_environment.py`

**Pre-flight:**
```
grep -rn "Kernel(" abides-* notebooks   # confirm all use kwargs
grep -rn "write_log\|write_summary_log\|log_dir" abides-* notebooks
grep -rn "kernel\.gym_agents\|update_raw_state\|apply_actions" abides-*
```

**Steps:**

1. **`LogWriter` Protocol.** Create `log_writer.py`:
   ```python
   from typing import Protocol
   import os, bz2, pickle, pandas as pd

   class LogWriter(Protocol):
       def write_agent_log(self, agent_name: str, df: pd.DataFrame) -> None: ...
       def write_summary_log(self, df: pd.DataFrame) -> None: ...

   class NullLogWriter:
       def write_agent_log(self, agent_name, df): pass
       def write_summary_log(self, df): pass

   class BZ2PickleLogWriter:
       def __init__(self, root: str | os.PathLike, run_id: str) -> None:
           self._root = os.path.abspath(root)
           self._run_id = run_id
           os.makedirs(os.path.join(self._root, run_id), exist_ok=True)
       def write_agent_log(self, agent_name, df):
           path = os.path.join(self._root, self._run_id, f"{agent_name}.bz2")
           with bz2.open(path, "wb") as f:
               pickle.dump(df, f)
       def write_summary_log(self, df):
           path = os.path.join(self._root, self._run_id, "summary_log.bz2")
           with bz2.open(path, "wb") as f:
               pickle.dump(df, f)
   ```
   Kernel takes `log_writer: LogWriter | None = None` keyword. If
   `None`, build `NullLogWriter() if skip_log else BZ2PickleLogWriter(log_root, log_dir)`.

2. **`log_root` parameter.**
   Add `log_root: str | os.PathLike = "./log"` to `Kernel.__init__`.
   Pass to `BZ2PickleLogWriter`. Resolve to absolute path at
   construction time.

3. **`KernelState` lifecycle enum.** Create `lifecycle.py`:
   ```python
   from enum import Enum, auto

   class KernelState(Enum):
       CONSTRUCTED = auto()
       INITIALIZED = auto()
       RUNNING = auto()
       STOPPED = auto()
       TERMINATED = auto()
   ```
   ```python
   from typing import Any, Protocol

   class GymAdapter(Protocol):
       def update_raw_state(self) -> None: ...
       def get_raw_state(self) -> Any: ...
       def apply_actions(self, actions: list[dict[str, Any]]) -> None: ...
   ```
   In `Kernel.__init__`:
   ```python
   def __init__(self, agents, *, gym_adapter: GymAdapter | None = None, ...):
       ...
       if gym_adapter is None:
           # Backwards-compat: auto-detect ABIDES-gym agents.
           detected = [a for a in agents
                       if "CoreGymAgent" in [c.__name__ for c in a.__class__.__bases__]]
           if detected:
               warnings.warn(
                   "Auto-detection of gym agents from `agents` is deprecated. "
                   "Pass gym_adapter=... explicitly.",
                   DeprecationWarning, stacklevel=2,
               )
               if len(detected) > 1:
                   raise ValueError("At most one gym agent allowed")
               gym_adapter = detected[0]
       self._gym_adapter = gym_adapter
   ```
   Replace every `self.gym_agents[0]` reference in `kernel.py` with
   `self._gym_adapter` (and check `if self._gym_adapter is not None`).

   In `abides-gym/abides_gym/envs/core_environment.py`, find the
   `Kernel(...)` construction and add `gym_adapter=<the gym agent>`.
   The gym agent already exists in the `agents` list — pass the
   reference. Public env API (observations, actions, reward) is
   unchanged.

5. **Keyword-only `__init__`.**
   ```python
   def __init__(
       self,
       agents: list[Agent],
       *,
       start_time: NanosecondTime = _DEFAULT_START_TIME,
       stop_time: NanosecondTime = _DEFAULT_STOP_TIME,
       ...
   ) -> None:
   ```
   Pre-flight grep confirmed all callers use kwargs — this is mechanical.

**Tests:**

- `test_null_log_writer_creates_no_files`
- `test_bz2_log_writer_round_trip`
- `test_kernel_uses_injected_log_writer`
- `test_lifecycle_rejects_out_of_order_initialize`
- `test_lifecycle_rejects_terminate_before_initialize`
- `test_gym_adapter_explicit_registration`
- `test_gym_auto_detection_emits_deprecation_warning`
- `test_kernel_init_keyword_only`

**Acceptance:**
- `pytest abides-core/tests/ abides-markets/tests/ abides-gym/ -x -q` green.
- A gym smoke test runs end-to-end (one episode of the gym env).

**Risk:** Medium-high. Touches abides-gym. Keep auto-detection
working until the next release.

---

## 2. Files Touched (consolidated)

| File | PRs |
|---|---|
| `abides-core/abides_core/kernel.py` | 1, 2, 3, 4, 5a, 6, 7 |
| `abides-core/abides_core/message.py` | 3 |
| `abides-core/abides_core/agent.py` | 4 |
| `abides-core/abides_core/latency_model.py` | 5a, 5b |
| `abides-core/abides_core/log_writer.py` | 7 (new) |
| `abides-core/abides_core/lifecycle.py` | 7 (new) |
| `abides-core/abides_core/gym_adapter.py` | 7 (new) |
| `abides-core/abides_core/abides.py` | 2, 7 |
| `abides-core/tests/test_kernel.py` | every PR |
| `abides-markets/abides_markets/agents/trading_agent.py` | 4 |
| `abides-markets/tests/test_market_boundaries.py` | 4 |
| `abides-markets/tests/test_seed_replicability.py` | 5b (baseline regen) |
| `abides-gym/abides_gym/envs/core_environment.py` | 7 |
| `docs/reference/llm-gotchas.md` | every PR (per-PR doc updates) |
| `docs/reference/kernel-architecture.md` | 7 (new — full architecture writeup) |
| `CHANGELOG.md` | every PR (5b is BREAKING) |

### Doc-update policy

Each PR includes its own documentation diff in the same commit set. No
trailing "docs catch-up" PR. Specifically:

- **PR 1:** CHANGELOG entry ("Quick wins").
- **PR 2:** `llm-gotchas.md` — add `initialize()` reset semantics; CHANGELOG.
- **PR 3:** `llm-gotchas.md` — add `_HeapEntry` ordering note and
  `WakeupMsg` singleton note; CHANGELOG.
- **PR 4:** `llm-gotchas.md` — replace `mean_result_by_agent_type`
  references with `report_metric()`; CHANGELOG.
- **PR 5a:** `llm-gotchas.md` — `LatencyModel.get_latency` keyword
  `random_state` arg; CHANGELOG.
- **PR 5b:** `llm-gotchas.md` — reproducibility-break section;
  **CHANGELOG with explicit BREAKING marker**.
- **PR 6:** `llm-gotchas.md` — `agent_current_times` is numpy `int64`;
  callers expecting Python `int` must cast; CHANGELOG.
- **PR 7:** `kernel-architecture.md` (new). `llm-gotchas.md` updates
  for `LogWriter`, `KernelState`, `GymAdapter`. CHANGELOG.

---

## 3. Out of Scope

- Per-agent FIFO queues in place of the in-future heap requeue (needs profiling first).
- Removing `kernel_starting` / `kernel_initializing` agent callbacks (public contract).
- Heap → calendar queue / skip list (premature without profiling).
- Multi-threaded dispatch (incompatible with discrete-event semantics).
- Migrating `custom_properties` → typed `kernel.extras` namespace
  (significant follow-up; touches markets and config system; deserves its own plan).
