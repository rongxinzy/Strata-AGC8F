# AGC8F expert parallel engineering branch

`feature/agc8f-expert-parallel` integrates the opt-in 24-layer candidate onto upstream `99f3dbd0b21d1401b3769e0c0d963913607f380b`. It is opt-in and not release-ready. The 48-layer experiment is excluded.

## Measured provenance

Base e1fe89ba93ab9b7f53b65f0aa1261a903c9561d8, source archive aaab89d3e02e4b9ef47e8dc1011a2d338a1697a88ce0b623e5c75e0b597e4376, engine 246224e179b11e3f0e6066a93358ea049487d636ef138b67a38e2eec0e1bed5a. The annotated tag `agc8f-ep24-measured-20261004` targets `878914090fe2c16bfcfd0432b2cdc9eb68519e00`; its five engine source files match the measured archive. The rebased branch includes upstream engine changes and no longer has those identical five files. All numbers below belong only to that pre-rebase source and binary. CUDA build, device correctness and performance on the new upstream base remain unverified.

Four independent boots OFF/ON/ON/OFF, 32 requests including 8 performance requests (two fixed prompts), IQ3_S/original MTP/FP16, 256 generated tokens and zero prompt reuse: decode throughput 58.42→60.76 tok/s at 1K and52.90→55.06 at16K, +4.01%/+4.08%. Both mirrored pairs exceed3.8%, OFF drift below0.24%. See AGC8F_EP24_SCREEN.json. Corresponding final text/draft counts match; QA remains4/5 with the known arithmetic error. End-to-end1K is inconsistent; these are limited decode screens, not broad performance or numerical acceptance. No full-logit equivalence claim.

## Candidate configuration and boundaries

Use exact `STRATA_EP_STAGE_TOP3=1` with 8 ordered GPUs, split6,12,18,24,30,36,42, `STRATA_STAGE_DENSE_SCOPE=1`, `STRATA_VERIFY_DEVICE_PLAN=0`, `STRATA_PREFILL_PIPELINE=1`, `STRATA_MTP_BATCH_MULTI=0`, prefill2048, spec4/min-p0.5, workers15 andCPU0–15, FP16 KV, adapt-swaps0, no-spec-split and no cache reuse. Seven helper caches add2.28–3.20GiB per GPU1–7; actual minimum free-memory checks remain mandatory. This description is not a portable launch command; recorded launch.json artifacts contain exact local paths and commands.

`STRATA_EP_L0`, `STRATA_EP_AUDIT`, and `STRATA_EP_TEST_SEQUENCE` are research hooks, not stable public configuration. Unset audit/test sequence for timings (AUDIT is presence-based). Default paths remain off. Host tests use CUDA/cache stubs and cannot establish device correctness.

## Upstream maintenance

Before each engineering/device round, fetch origin and upstream, record upstream/main SHA and divergence. Preserve a measured tag before rewriting branch history. Rebase onto upstream/main in a bounded integration step; resolve semantics rather than copying whole old files over upstream changes. Use force-with-lease only for this owned feature branch after review. Do not rewrite upstream-mirror main or the agc8f validated branch. A rebase invalidates prior binary performance attribution: rebuild independently and repeat numeric/QA and paired timing checks before quoting new-version gains.

EP explicitly rejects every nonnegative `--peer-device`, including device 0, before helper setup. Upstream peer-cache and peer-prefill paths remain available with EP disabled. EP helper input, output and metadata staging retain explicit `cudaHostAllocPortable`; they do not depend on upstream's peer-only portable setting. This source inspection and host regression coverage do not establish device compatibility.

## Remaining engineering acceptance

Replace temporary hooks with validated configuration, audit allocation/error/teardown paths, add real-device cancellation/restart checks, expand held-out workloads and only then consider merging into agc8f with the feature still default-off. Keep the measured24-layer baseline distinct from48-layer exploration.
