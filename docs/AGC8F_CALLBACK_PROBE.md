# AGC8F callback state diagnostic

The previous end-of-request state audit could not tell whether an MTP callback
changed main state or whether main prefill had already diverged. This opt-in probe
reads the same owner/device ranges before the callback, immediately again, and
after the callback. It never writes device memory.

Enable `STRATA_MAIN_CALLBACK_PROBE=1` and set
`STRATA_MAIN_CALLBACK_PROBE_DIR` to a new directory whose parent already exists.
An existing destination is rejected. The default path performs no CUDA reads or
file output. Each chunk preserves raw FP32, hashes, finite-value statistics,
bit differences, active recurrence/conv boundaries, actual path tags and address
overlap checks. Actual device addresses belong in local evidence, not public logs.

## Device experiment, 2026-10-03

Hardware: agc8f, eight RTX 4060 Ti 16GB, Hygon 16 physical cores. Main weights:
GSQ-RCO IQ3_S; original MTP and FP16 KV unchanged. This is a numerical diagnostic,
not a performance measurement or a quality pass.

Tested implementation: `49c18b2336b178740792aff7a71256dee2ed5e6f`.
CUDA Release, sm89 binary SHA-256:
`1ffcfaef1bddd80d5607ecee6cad82e88da7aecdf1ac644cf8ebd916965295c9`.
Later commits add host tests and documentation without changing the device hook.

Four independent boots used `STRATA_MTP_BATCH_MULTI=0,1,1,0`. Every boot ran
one warmup and three prompts of 2112/2113/2114 tokens, producing seven callbacks.
All four boots enabled the probe, snapshot audit and prefill pipeline audit;
pipeline=1, prefill=2048, device-plan=0, spec=4, min-p=0.5. These audited runs are
excluded from timing comparisons. The warmup prefills 1855 rows of a 1856-token
prompt; the last prompt token is handled separately by the serving path.

An independent evaluator recomputed all 252 raw samples, including hashes,
statistics, path/request correspondence and runtime address overlaps:

| Observation | Result |
| --- | --- |
| Before vs immediate reread | 84/84 ranges bitwise identical |
| Before vs after callback | 84/84 ranges bitwise identical |
| Raw/metadata or association errors | 0 |
| Nonfinite FP32 samples | 0 |
| Boot 1 vs boot 2, 3 or 4, before callback | 21/21 ranges differ in each pair |
| Boot 2 vs 3, 2 vs 4, 3 vs 4 | 21/21 ranges identical in each pair |

Boot 1 differs from another boot with batching disabled. Its first warmup
callback already has different **before** state. Therefore these data do not
attribute the differences to the MTP batching switch. Covered ranges undergo no
net bit change during the instrumented callback.

The four boots used the same messages, target prompt IDs, seed 42, temperature 0,
top-k 1 and no prompt reuse. Actual prompt counts were checked; this round did
not independently capture every boot's tokenizer IDs. Boot 1's warmup and
2114-token prompt produced different text from boots 2–4; the other two tail
prompts matched. Boots 2–4 matched text throughout. Text equality is not inferred
from state equality, and the differences are retained.

For boot 1 versus boot 2 across the seven callbacks:

| Range | Maximum absolute FP32 difference | Relative L2 difference, min–max |
| --- | ---: | ---: |
| GDN45 | 0.00003030058 | 1.659e-7–1.559e-6 |
| GDN46 | 0.02384186 | 2.914e-4–7.717e-4 |
| Last residual row | 0.01130009 | 1.103e-3–3.649e-3 |

The relative norms above include each entire listed range, including the conv
suffix for GDN. GDN45 differs only in recurrence; its conv suffix is identical. GDN46 differs
in recurrence and conv. This localizes a difference, not its first originating
layer or its cause. Relative L2 uses the first sample as denominator.

## Boundaries and validation

Coverage is last-stage GDN45/46 and the last active residual row. QSA KV, other
layers, other residual rows and later decode state are not covered.
`on_stage_chunk` executes before the first sample. Device-wide synchronization
and file I/O alter scheduling and can mask a race. Equal sampled state therefore
does not prove the uninstrumented engine is race-free. A first-boot effect, a
particular kernel and output quality remain unresolved; the older sanity set
still has its documented arithmetic failure.

The actual draft path is independently checked against engine/MTP audit records;
`not_called` or `declined` alone does not establish a completed token fallback.
Host regression covers disabled mode, missing directory, overwrite rejection,
read failure, bounded 64KiB reads, signed zero, nonfinite values and conv changes.
These host tests validate the diagnostic helper, not GPU arithmetic. CUDA build
and the existing three device-build CTests passed; five host cases also run in CI.

Raw evidence and the independent evaluator are retained in the research workspace
at `autoresearch/agc8f/iterations/001/`, rather than adding large model-derived
state dumps to this repository. The next investigation is main-state divergence
before the first callback, with cold/warm and synchronization effects still to
be distinguished.
