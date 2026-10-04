# AGC8F expert parallel engineering branch

`feature/agc8f-expert-parallel` integrates the opt-in 24-layer candidate onto upstream `99f3dbd0b21d1401b3769e0c0d963913607f380b`. It is opt-in and not release-ready. The 48-layer experiment is excluded.

## Measured provenance

Base e1fe89ba93ab9b7f53b65f0aa1261a903c9561d8, source archive aaab89d3e02e4b9ef47e8dc1011a2d338a1697a88ce0b623e5c75e0b597e4376, engine 246224e179b11e3f0e6066a93358ea049487d636ef138b67a38e2eec0e1bed5a. The annotated tag `agc8f-ep24-measured-20261004` targets `878914090fe2c16bfcfd0432b2cdc9eb68519e00`; its five engine source files match the measured archive. The rebased branch includes upstream engine changes and no longer has those identical five files. All numbers below belong only to that pre-rebase source and binary. The rebased code at `45b246075f489eb47b367067dfdba59560cb9a01` passed an independent CUDA sm89 Release build and 26 CTest checks on agc8f. The new-base limited replication is recorded below; full numerical and broad performance acceptance remain outstanding.

Four independent boots OFF/ON/ON/OFF, 32 requests including 8 performance requests (two fixed prompts), IQ3_S/original MTP/FP16, 256 generated tokens and zero prompt reuse: decode throughput 58.42→60.76 tok/s at 1K and52.90→55.06 at16K, +4.01%/+4.08%. Both mirrored pairs exceed3.8%, OFF drift below0.24%. See AGC8F_EP24_SCREEN.json. Corresponding final text/draft counts match; QA remains4/5 with the known arithmetic error. End-to-end1K is inconsistent; these are limited decode screens, not broad performance or numerical acceptance. No full-logit equivalence claim.

## Candidate configuration and boundaries

Use exact `STRATA_EP_STAGE_TOP3=1` with 8 ordered GPUs, split6,12,18,24,30,36,42, `STRATA_STAGE_DENSE_SCOPE=1`, `STRATA_VERIFY_DEVICE_PLAN=0`, `STRATA_PREFILL_PIPELINE=1`, `STRATA_MTP_BATCH_MULTI=0`, prefill2048, spec4/min-p0.5, workers15 andCPU0–15, FP16 KV, adapt-swaps0, no-spec-split and no cache reuse. Seven helper caches add2.28–3.20GiB per GPU1–7; actual minimum free-memory checks remain mandatory. This description is not a portable launch command; recorded launch.json artifacts contain exact local paths and commands.

`STRATA_EP_L0`, `STRATA_EP_AUDIT`, and `STRATA_EP_TEST_SEQUENCE` are research hooks, not stable public configuration. Unset audit/test sequence for timings (AUDIT is presence-based). Default paths remain off. Host tests use CUDA/cache stubs and cannot establish device correctness.

## Upstream maintenance

Before each engineering/device round, fetch origin and upstream, record upstream/main SHA and divergence. Preserve a measured tag before rewriting branch history. Rebase onto upstream/main in a bounded integration step; resolve semantics rather than copying whole old files over upstream changes. Use force-with-lease only for this owned feature branch after review. Do not rewrite upstream-mirror main or the agc8f validated branch. A rebase invalidates prior binary performance attribution: rebuild independently and repeat numeric/QA and paired timing checks before quoting new-version gains.

EP explicitly rejects every nonnegative `--peer-device`, including device 0, before helper setup. Upstream peer-cache and peer-prefill paths remain available with EP disabled. EP helper input, output and metadata staging retain explicit `cudaHostAllocPortable`; they do not depend on upstream's peer-only portable setting. This source inspection and host regression coverage do not establish device compatibility.

## Remaining engineering acceptance

Configuration validation, allocation/error handling and real-device cancellation checks are completed below. Keep research hooks explicitly separate. Broader fresh held-out workloads, full internal numerical comparison and prolonged stress remain future acceptance work before any default enablement. Keep the measured24-layer baseline distinct from48-layer exploration.

## Rebased build receipt (2026-10-04)

Source archive `bad81088d143e8a76facbf20b18d7a4aa8c37a40d105450182de7a314436d7e0`, engine SHA256 `0bf32c52f8057b389c1d9453194b15c0e3afdce95d7013e40a77c84e67f56305`, CUDA13.2/sm89, fixed ggml `3cf03257f219afbe7334045ff7c6a06ac68c627d`. Independent engine directory `Strata-AGC8F-ep24-rebased-213-v1`; no model request or new speed measurement was performed. Build completed in117.8s, 26/26 CTest passed, devices idle and ports closed afterward. Local EP plain/ASan/UBSan controls also passed; they do not replace real-device validation.

## Latest-upstream device replication (2026-10-04, 214/215)

Using the above rebased binary, four independent OFF/ON/ON/OFF boots completed32 requests, including8 performance requests. This compares the same rebased fork with EP24 disabled/enabled, not an unmodified upstream binary. Other AGC8F patches and settings are identical. Upstream was fetched again and remained `99f3dbd`.

| Input tokens | OFF decode tok/s | ON decode tok/s | Throughput gain | Mirrored pair gains | OFF elapsed drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | 57.83 | 60.91 | 5.32% | 6.94% / 3.70% | 2.93% |
| 16384 | 53.08 | 55.14 | 3.88% | 3.94% / 3.82% | 0.05% |

The approximately4% decode signal survives. The noisier1K OFF baseline does not establish a stable5% gain. Pooled client elapsed time decreases5.82%/1.10% in these two cases; these are separate from decode throughput. All corresponding final texts, output counts and draft/accepted counts match across the four boots. Performance requests each generated256 tokens with zero cache reuse. QA remains4/5 with the known arithmetic error. Two fixed prompts with two observations per arm are not a broad held-out test or full internal numerical proof. The feature remains opt-in.

`AGC8F_EP24_UPSTREAM_SCREEN.json` preserves timings, provenance, raw record hashes and the independent audit summary. Full raw artifacts remain in workspace `autoresearch/agc8f/iterations/214-ep24-upstream-abba/raw-gate`; each boot retains exact launch commands and actual environment/runtime hashes. Independent reviewer215 and a separate root recomputation agree. All owned processes exited, both localhost ports closed, all GPUs idle and the global lock released. Old measured-tag results remain separate and are not added to these gains.

## Configuration and failure behavior

The supported opt-in remains `STRATA_EP_STAGE_TOP3=1`; leave it unset or set `0` to use the original path. `STRATA_EP_STAGE_TOP3` and the research-only `STRATA_EP_L0` now accept only exact `0`/`1`; empty strings, `true`, `01`, and whitespace are rejected instead of silently disabling a requested mode. They are mutually exclusive. The existing fixed topology/precision/residency guards still apply. Research audit/sequence hooks remain explicitly separate and must be unset for deployment and performance measurements.

Helper initialization publishes its expert ownership only after all allocations and the final capacity check succeed. Standard initialization exceptions close partial resources and return an error. Invalid windows and null input pointers are rejected before size multiplication. An incomplete asynchronous submission retains its pending lease until drained and cannot publish stale result rows. Synchronization failure retains the lease; another request cannot overwrite its staging. These changes preserve the measured expert assignment and normal kernel arithmetic.

Host tests inject allocation, copy, launch and synchronization failures using CUDA stubs. They do not establish recovery from a real sticky CUDA error or device loss; those require terminating the affected process and a healthy device before restarting. No driver reset or device-loss injection is performed by this project.

## Lifecycle device evidence (219 build / 223 run)

The independent sm89 build of the closeout source passed26 CTest checks. An audit-only device run disconnected decode after65 visible engine tokens and an actual EP_SERVE witness; the engine stopped at67 generated tokens, below the2048 request budget. A16K prompt cancellation stopped after2048 prompt tokens with zero generated output. The same engine PID and birth timestamp then completed the normal eight-request warmup/QA/validation sequence. API-parent TERM completed with exit0 and no harness KILL; ports closed, all eight GPUs idle, lock released. API close retains its own escalation policy; this is not a per-destructor trace or proof of device-loss recovery.

The earlier217 cancellation attempt is retained: it cancelled decode before the EP path had run, and one group-TERM shutdown required KILL. It is excluded from EP-active cancellation and cooperative-shutdown acceptance. Two inherited223 controller metadata fields (a self-comparison and a RESTART label for a single boot) are also excluded; acceptance uses raw request, PID, phase, EP-log and cleanup evidence. Audit timings are never performance evidence.

## Final closeout matrix (220, source889028a)

The final lifecycle-fixed source at `889028a755e067d96f16d857043fe91d1376ae8e` is byte-identical to the219 build for all include/src/tools-agc8f files. Engine SHA256 `e16d52baf233bd52230174a0d9a133d2b1812a09eec194f7424664df083c869d`. Source CI37200514902 passed. Four independent OFF/ON/ON/OFF boots completed60 requests, including36 performance requests: Chinese/English/code at1K/4K/16K,256 output tokens and zero prompt reuse, no audit/profile/test sequence. These are existing frozen validation inputs, not newly held-out data. The16K Chinese prompt has16383 actual tokens; the other two have16384.

| Input target | OFF decode tok/s | ON decode tok/s | Throughput gain | Client elapsed reduction |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 61.67 | 64.31 | 4.28% | 4.68% |
| 4096 | 60.40 | 62.85 | 4.05% | 2.27% |
| 16384 | 58.26 | 60.79 | 4.34% | 2.70% |

Across all36 performance requests, decode60.08→62.62tok/s (+4.22%; mirrored pairs+4.30%/+4.15%; OFF elapsed drift0.094%). Client cumulative elapsed decreases3.06%. Every individual input improves in both mirrored pairs. Corresponding final text/output/draft/accepted counts match across all four boots; QA retains the known4/5 result. Text equality is not full-logit/internal-state proof, and this limited matrix does not establish p95 or general workload guarantees. Four sequential server boots all completed API-parent TERM with exit0, no harness KILL, and idle cleanup. The feature remains default-off.

See `AGC8F_EP24_FINAL_MATRIX.json` for raw-derived timings/provenance/hashes. Workspace raw: `autoresearch/agc8f/iterations/220-ep24-final-matrix/raw-gate`, including exact launch commands. Independent222 reviewed lifecycle raw and matrix formulas; the complete raw checker and a separate root recomputation agree. Prior measured-tag and214 results are retained separately.
