# ABIDES-Core

**Core discrete-event simulation engine for agent-based modeling.**

ABIDES-Core provides the foundational infrastructure for multi-agent
discrete-event simulations. It implements an event-driven kernel, a
latency-aware messaging system, and a base `Agent` class that all
domain-specific agents extend. The module is intentionally
domain-agnostic: financial markets, network protocols, or any system
whose dynamics emerge from asynchronous agent interactions can be
modeled on top of it.

## Architecture

| Component | Description |
|-----------|-------------|
| **Kernel** | Priority-queue event loop that drives simulation time forward, dispatches messages and wakeups, and enforces per-agent computation delays. |
| **Agent** | Abstract base class with a well-defined lifecycle (`kernel_initializing` → `kernel_starting` → `wakeup` / `receive_message` → `kernel_stopping` → `kernel_terminating`). |
| **Message** | Lightweight dataclass. Delivery metadata (sender, recipient, timestamp) is managed by the Kernel, not embedded in the message. `MessageBatch` applies computation delay once per batch for high-frequency agents. |
| **LatencyModel** | Stochastic network latency model. Default *cubic* mode produces realistic heavy-tailed distributions; a *deterministic* mode is available for controlled experiments. Supports scalar, per-agent, or pairwise (N×N) parameterization. |

### Timing conventions

All timestamps are **64-bit nanoseconds** (Unix epoch), aliased as
`NanosecondTime = int`. Prices, when used in downstream modules, are
**integer cents** (`$100.00 = 10_000`).

## Public API

```python
from abides_core import Agent, Kernel, LatencyModel, Message, NanosecondTime
```

## Agent lifecycle

Agents are reactive: the only entry points are `wakeup()` and
`receive_message()`. There are no polling loops.

```
kernel_initializing(kernel)   # obtain kernel reference; no inter-agent comms
kernel_starting(start_time)   # all agents exist; request first wakeup
wakeup(current_time)          # scheduled alarm fires
receive_message(current_time, sender_id, msg)  # incoming message delivered
kernel_stopping()             # simulation ending; final inter-agent comms
kernel_terminating()          # cleanup; write logs
```

### Key agent methods

```python
send_message(recipient_id, message, delay=0)
send_message_batch(recipient_id, messages, delay=0)
set_wakeup(requested_time)
set_computation_delay(ns)
logEvent(event_type, event)
```

## Message delivery pipeline

When an agent calls `send_message`:

1. **Compute send time** — `current_time + computation_delay + additional_delay + delay`
2. **Apply network latency** — sampled from the `LatencyModel` for the `(sender, recipient)` pair
3. **Enqueue** — the message is pushed onto the kernel's priority queue keyed by delivery time
4. **Agent becomes busy** — the sender cannot process new events until its computation delay elapses

## Latency model

```python
LatencyModel(
    random_state,
    min_latency,                # 2-D ndarray [sender, recipient] in ns
    latency_model="cubic",      # or "deterministic"
    jitter=0.5,                 # cubic curve shape (0–1)
    jitter_clip=0.1,            # clip distribution tail
    jitter_unit=10,             # scaling factor
    connected=True,             # bool or ndarray to block pairs
)
```

## Running a simulation

```python
from abides_core import abides

# `config_state` is a dict produced by a config builder
# (see abides-markets for the declarative config system).
end_state = abides.run(config_state)
```

A CLI entry point is also provided:

```bash
abides path/to/config.py [--param value ...]
```
