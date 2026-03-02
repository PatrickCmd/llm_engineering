# Multiprocessing and Retry in Python — A Practical Tutorial

This tutorial covers two techniques used in our document ingestion pipeline (`ingest.py`):

1. **`tenacity`** — a retry library that automatically re-attempts failed function calls with configurable backoff strategies.
2. **`multiprocessing`** — a standard-library module that runs CPU- or IO-bound work across multiple OS processes in parallel.

Both are general-purpose tools, but they pair especially well in pipelines that call external APIs (like LLMs) where requests can fail transiently and throughput benefits from parallelism.

---

## Part 1: Retry with `tenacity`

### 1.1 The problem

External API calls fail. Networks drop, rate limits kick in, servers return 500s. Without retry logic your pipeline crashes on the first transient error, even though the same request would succeed a few seconds later.

You *could* write a manual retry loop:

```python
import time

def call_api(payload):
    for attempt in range(5):
        try:
            return api.send(payload)
        except Exception:
            time.sleep(2 ** attempt)
    raise RuntimeError("All retries exhausted")
```

This works but quickly gets messy when you need different strategies for different functions, want to log attempts, or need to retry only on certain exceptions. `tenacity` solves this declaratively.

### 1.2 Installation

```bash
pip install tenacity
```

### 1.3 Basic usage — the `@retry` decorator

The simplest form retries forever until the function succeeds:

```python
from tenacity import retry

@retry
def unreliable():
    result = call_flaky_api()
    return result
```

This will keep retrying with no delay and no limit — useful for demos, but dangerous in production. You almost always want to add a **wait strategy** and a **stop condition**.

### 1.4 Wait strategies

`tenacity` provides several built-in wait strategies. Each controls *how long* to pause between attempts.

#### Fixed wait

Pause the same amount of time every retry:

```python
from tenacity import retry, wait_fixed

@retry(wait=wait_fixed(2))
def fetch_data():
    return requests.get("https://api.example.com/data").json()
```

Timeline: attempt → wait 2s → attempt → wait 2s → …

#### Exponential backoff

Each wait is longer than the last — the standard approach for rate-limited APIs:

```python
from tenacity import retry, wait_exponential

@retry(wait=wait_exponential(multiplier=1, min=2, max=60))
def fetch_data():
    return requests.get("https://api.example.com/data").json()
```

Timeline: attempt → wait 2s → attempt → wait 4s → attempt → wait 8s → … (capped at 60s).

The parameters:

| Parameter | Meaning |
|-----------|---------|
| `multiplier` | Scales the base wait. `multiplier=1` means the raw exponential value is used. |
| `min` | Minimum wait in seconds (floor). |
| `max` | Maximum wait in seconds (ceiling). |

The formula is: `wait = min(max, max(min, multiplier × 2 ** attempt_number))`.

This is exactly what our ingestion pipeline uses:

```python
wait = wait_exponential(multiplier=1, min=10, max=240)

@retry(wait=wait)
def process_document(document):
    ...
```

So the first retry waits 10s, the next 20s, then 40s, 80s, 160s, and caps at 240s (4 minutes). This gives an overloaded API plenty of breathing room.

#### Random jitter

Add randomness to prevent multiple workers from retrying in lockstep (the "thundering herd" problem):

```python
from tenacity import retry, wait_exponential_jitter

@retry(wait=wait_exponential_jitter(initial=1, max=60, jitter=5))
def fetch_data():
    return requests.get("https://api.example.com/data").json()
```

Each wait is the exponential value plus a random offset between 0 and `jitter` seconds.

### 1.5 Stop conditions

Without a stop condition, retries continue forever. You can limit by count or total time.

#### Stop after N attempts

```python
from tenacity import retry, stop_after_attempt, wait_fixed

@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
def fetch_data():
    return requests.get("https://api.example.com/data").json()
```

After 5 failed attempts, the original exception is re-raised.

#### Stop after a time limit

```python
from tenacity import retry, stop_after_delay, wait_fixed

@retry(stop=stop_after_delay(120), wait=wait_fixed(5))
def fetch_data():
    return requests.get("https://api.example.com/data").json()
```

Gives up after 120 seconds total, regardless of how many attempts were made.

#### Combine stop conditions

```python
from tenacity import retry, stop_after_attempt, stop_after_delay, wait_exponential

@retry(
    stop=(stop_after_attempt(10) | stop_after_delay(300)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
)
def fetch_data():
    ...
```

Stops after 10 attempts **or** 5 minutes, whichever comes first.

### 1.6 Retrying only on certain exceptions

By default `tenacity` retries on *any* exception. You can narrow this:

```python
from tenacity import retry, retry_if_exception_type, wait_exponential

@retry(
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)
def fetch_data():
    return requests.get("https://api.example.com/data", timeout=10).json()
```

A `ValueError` would *not* trigger a retry — it would propagate immediately.

### 1.7 Logging retries

Use the `before_sleep` callback to log each retry:

```python
from tenacity import retry, wait_exponential, before_sleep_log
import logging

logger = logging.getLogger(__name__)

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def fetch_data():
    return requests.get("https://api.example.com/data").json()
```

Each retry logs a warning like: `Retrying fetch_data in 4.0 seconds as it raised ConnectionError: ...`

### 1.8 Accessing retry statistics

After a call completes, `tenacity` attaches statistics to the function:

```python
result = fetch_data()
print(fetch_data.retry.statistics)
# {'start_time': 1709..., 'attempt_number': 3, 'idle_for': 6.0}
```

### 1.9 Complete example — retrying an LLM call

```python
from tenacity import retry, wait_exponential, stop_after_attempt
from litellm import completion

@retry(
    wait=wait_exponential(multiplier=1, min=10, max=240),
    stop=stop_after_attempt(6),
)
def summarize(text: str) -> str:
    response = completion(
        model="openai/gpt-4.1-mini",
        messages=[{"role": "user", "content": f"Summarize:\n\n{text}"}],
    )
    return response.choices[0].message.content
```

If the OpenAI API returns a rate-limit error (429), `tenacity` waits 10s, then 20s, then 40s, up to 240s, for a maximum of 6 attempts before giving up.

---

## Part 2: Multiprocessing with `multiprocessing.Pool`

### 2.1 The problem

Python's Global Interpreter Lock (GIL) means threads cannot run Python bytecode in true parallel. For CPU-bound work this is a hard limit. For IO-bound work (like API calls), threads *can* help, but processes give you full isolation and avoid GIL contention entirely.

`multiprocessing` spawns separate OS processes, each with its own Python interpreter and memory space. Work is distributed across these processes and results are collected back in the parent.

### 2.2 Key concepts

| Concept | Description |
|---------|-------------|
| **Process** | An independent OS process with its own memory and Python interpreter. |
| **Pool** | A managed group of worker processes. You submit tasks; the pool assigns them to available workers. |
| **Worker** | One process in the pool. |
| **Serialization (pickling)** | Arguments and return values are serialized to pass between processes. Functions and data must be *picklable*. |

### 2.3 `Pool` — the workhorse

`multiprocessing.Pool` is the most common entry point. It manages a fixed number of worker processes and provides several methods to distribute work.

#### Basic pattern

```python
from multiprocessing import Pool

def square(x):
    return x * x

with Pool(processes=4) as pool:
    results = pool.map(square, [1, 2, 3, 4, 5])

print(results)  # [1, 4, 9, 16, 25]
```

The `with` statement ensures workers are properly cleaned up when done.

### 2.4 `pool.map` vs `pool.imap` vs `pool.imap_unordered`

These three methods differ in *when* and *how* results are returned:

#### `pool.map(func, iterable)`

Blocks until **all** items are processed, then returns a list in input order.

```python
with Pool(4) as pool:
    results = pool.map(process_item, items)
# results is a complete list, same order as items
```

- **Pro**: simple, familiar (like built-in `map`).
- **Con**: you wait for the slowest item before getting *any* results. High memory usage for large iterables (entire input buffered).

#### `pool.imap(func, iterable)`

Returns an **iterator** that yields results in input order, as they become available.

```python
with Pool(4) as pool:
    for result in pool.imap(process_item, items):
        print(result)  # results arrive in input order
```

- **Pro**: start processing results before all items are done. Lower memory.
- **Con**: if item 3 is slow but items 4–10 are fast, you still wait for item 3 before seeing 4–10.

#### `pool.imap_unordered(func, iterable)`

Returns an **iterator** that yields results in **completion order** — whichever finishes first comes out first.

```python
with Pool(4) as pool:
    for result in pool.imap_unordered(process_item, items):
        print(result)  # fastest results come first
```

- **Pro**: maximum throughput. No head-of-line blocking. Ideal for progress bars.
- **Con**: results are in arbitrary order.

This is what our ingestion pipeline uses:

```python
def create_chunks(documents):
    chunks = []
    with Pool(processes=WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(process_document, documents), total=len(documents)):
            chunks.extend(result)
    return chunks
```

`imap_unordered` is chosen because:
1. We don't care about document order — we just need all chunks.
2. Some documents are longer than others, so completion times vary.
3. `tqdm` updates the progress bar as each document finishes, giving real-time feedback.

### 2.5 Choosing the number of workers

| Workload type | Guideline |
|---------------|-----------|
| **CPU-bound** (math, parsing, compression) | `os.cpu_count()` or slightly less |
| **IO-bound** (API calls, file reads) | Can exceed CPU count; limited by API rate limits |

Our pipeline sets `WORKERS = 5` because the bottleneck is the LLM API, not the CPU. With 5 workers we get up to 5 concurrent API calls. If you hit rate limits, reduce to 1.

### 2.6 Gotchas and common pitfalls

#### Functions must be defined at the module level

`Pool` serializes (pickles) the target function and sends it to worker processes. Lambdas, closures, and nested functions **cannot** be pickled:

```python
# This FAILS
with Pool(4) as pool:
    results = pool.map(lambda x: x * x, [1, 2, 3])
# PicklingError: Can't pickle <lambda>

# This WORKS
def square(x):
    return x * x

with Pool(4) as pool:
    results = pool.map(square, [1, 2, 3])
```

#### Arguments and return values must be picklable

Complex objects (database connections, open file handles, sockets) cannot be passed between processes. Keep arguments simple (strings, dicts, Pydantic models) and create connections *inside* the worker function.

#### Global state is not shared

Each worker process gets its own copy of global variables. Modifying a global in one worker does not affect the parent or other workers:

```python
counter = 0

def increment(_):
    global counter
    counter += 1
    return counter

with Pool(4) as pool:
    results = pool.map(increment, range(10))

print(results)   # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  — not [1,2,3,...,10]
print(counter)   # 0  — parent's counter is unchanged
```

If you need shared state, use `multiprocessing.Value`, `multiprocessing.Queue`, or `multiprocessing.Manager` — but prefer stateless designs.

#### Protect the entry point with `if __name__ == "__main__"`

On macOS and Windows, `multiprocessing` uses `spawn` to create new processes. Each worker re-imports your module. Without the guard, the pool creation itself gets re-executed in each worker, causing infinite recursion:

```python
# WRONG — infinite spawn loop on macOS/Windows
from multiprocessing import Pool

def work(x):
    return x * x

pool = Pool(4)                      # executed on import!
results = pool.map(work, range(10))

# CORRECT
from multiprocessing import Pool

def work(x):
    return x * x

if __name__ == "__main__":
    with Pool(4) as pool:
        results = pool.map(work, range(10))
```

### 2.7 Complete example — parallel file processing

```python
import json
from pathlib import Path
from multiprocessing import Pool

def process_file(filepath: str) -> dict:
    """Read a JSON file and return a summary."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return {
        "file": filepath,
        "keys": len(data),
        "size_bytes": Path(filepath).stat().st_size,
    }

if __name__ == "__main__":
    files = list(Path("data").glob("*.json"))

    with Pool(processes=4) as pool:
        summaries = pool.map(process_file, [str(f) for f in files])

    for s in summaries:
        print(f"{s['file']}: {s['keys']} keys, {s['size_bytes']} bytes")
```

---

## Part 3: Combining Multiprocessing and Retry

The real power comes from combining both: multiple workers process documents in parallel, and each worker automatically retries on transient failures.

### 3.1 How it works in our pipeline

```
┌─────────────────────────────────────────────┐
│              Parent Process                  │
│                                              │
│  documents ──→ Pool(WORKERS=5)               │
│                  │                           │
│         ┌───────┼───────┐───────┐───────┐    │
│         ▼       ▼       ▼       ▼       ▼    │
│      Worker1 Worker2 Worker3 Worker4 Worker5  │
│         │       │       │       │       │    │
│         ▼       ▼       ▼       ▼       ▼    │
│     process_  process_  ...    ...    ...     │
│     document  document                       │
│     (@retry)  (@retry)                       │
│         │       │                            │
│         ▼       ▼                            │
│      ┌──────────────────────┐                │
│      │  LLM API (litellm)  │                │
│      └──────────────────────┘                │
│                                              │
│  Results collected via imap_unordered + tqdm │
└─────────────────────────────────────────────┘
```

1. The parent process creates a pool of 5 workers.
2. Each worker picks up a document from the queue.
3. Inside `process_document`, the `@retry` decorator handles transient API failures with exponential backoff (10s → 20s → 40s → … → 240s).
4. A worker that is retrying does not block the others — they continue processing their own documents.
5. Results stream back to the parent via `imap_unordered`, updating the progress bar in real time.

### 3.2 Why this combination works well

| Concern | Solution |
|---------|----------|
| API calls are slow | Multiprocessing sends N requests concurrently |
| API calls fail transiently | `@retry` handles failures without crashing the pipeline |
| Rate limits trigger under load | Exponential backoff gives the API time to recover |
| Some documents take longer | `imap_unordered` avoids head-of-line blocking |
| Worker crashes | The pool detects it and raises the exception in the parent |

### 3.3 Adjusting for rate limits

If you start seeing frequent retries (all workers hitting rate limits simultaneously), reduce `WORKERS`:

```python
WORKERS = 1  # serialize all requests to stay within rate limits
```

Alternatively, add jitter to spread out retries:

```python
from tenacity import retry, wait_exponential_jitter

@retry(wait=wait_exponential_jitter(initial=10, max=240, jitter=5))
def process_document(document):
    ...
```

### 3.4 Multiprocessing vs. threading vs. asyncio

| Approach | Best for | GIL-free? | Overhead |
|----------|----------|-----------|----------|
| `multiprocessing` | CPU-bound work or mixed workloads | Yes (separate processes) | High (process creation, pickling) |
| `threading` | IO-bound work (network, disk) | No (GIL limits CPU parallelism) | Low (shared memory) |
| `asyncio` | High-concurrency IO (thousands of connections) | No (single thread, cooperative) | Very low (coroutines) |

Our pipeline uses `multiprocessing` because:
- Each worker makes a blocking API call and then parses the result (a mix of IO and CPU).
- Process isolation means a crash in one worker doesn't affect others.
- The number of concurrent workers is small (5), so the overhead of process creation is negligible.

For pipelines with hundreds of concurrent API calls and no CPU work, `asyncio` with `aiohttp` would be more efficient. For light IO with shared data structures, `threading` with `concurrent.futures.ThreadPoolExecutor` is simpler.
