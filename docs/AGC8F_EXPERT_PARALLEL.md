# AGC8F expert parallel

`feature/agc8f-expert-parallel` carries opt-in EP24 on upstream
`6f32ec070f23ced9f50e704d854d775da52591ab`. EP is **off by default**.
The current source includes the fresh-activation publication fix over `fe6fb61`.
Its independent sm89 build SHA-256 is
`6a5489b45d032e216cbc4e9f205709ffecf9a7005255887ca87e32a95e94bd64`.

## Current validation status

The complete build passed 26 CTest checks. The targeted publication test, linked
against its production CUDA library, passed 320 cases including consecutive
inputs and the unchanged OFF resident control. Independent raw review passed.
Real-model validation of this fix is running; no new speed result is accepted.

The preceding build's representative expert-chain CUDA
validation completed 56 cases over seven actual quantization pairs, including
42 active EP and 14 local-fallback cases. The raw checker passed 336 full-output
byte comparisons and 252 one-shot helper API rejection/recovery suites.
Independent raw acceptance confirmed all 2,241 files and 663,000,972 bytes
against the device artifacts. Current-model activity, lifecycle and performance
validation are in progress. These results do not establish full-model state or
logit equivalence, sticky CUDA-error recovery, or a current speed advantage.

The host CI includes remote resource lifetime, rank dispatch, pending-result
drain, prefill failure flow, verifier host-policy and activation-publication
regressions. Its dependency
stubs do not execute CUDA graphs or model mathematics.

## Why the upstream integration needs a dispatch policy

The new upstream automatically uses an all-resident zero-doorbell verifier
when all experts fit in the GPU caches. This skips the host pool callback where
EP dispatch is installed. The first integration initialized all seven helper
caches but produced no EP activity; that failed smoke run is retained.

The current repair requests host expert dispatch before every EP verifier is
initialized. It disables both all-resident and device-plan bypasses for EP.
OFF retains upstream's default all-resident path. ON uses host plans on all
48 layers and runs remote experts on the 24 selected layers. Thus ON pays a
host scheduling cost that the current OFF baseline avoids. The resident publication API previously skipped copying host activations even
though EP helpers needed them. EP now passes null residency to that API, forcing
publication of the current activation; expert arithmetic is unchanged. The
preceding model smoke exposed stale-input output differences and failed.
Historical gains
cannot be carried forward or established by disabling the default path in OFF.

## Supported configuration

Use exact `STRATA_EP_STAGE_TOP3=1` with eight ordered GPUs, stage cuts
`6,12,18,24,30,36,42`, `STRATA_STAGE_DENSE_SCOPE=1`,
`STRATA_VERIFY_DEVICE_PLAN=0`, `STRATA_PREFILL_PIPELINE=1`,
`STRATA_MTP_BATCH_MULTI=0`, prefill 2048, spec 4/min-p 0.5, workers 15,
CPU 0–15, original IQ3_S/MTP, FP16 KV, adapt-swaps 0 and no-spec-split.
Keep audit/profile/test hooks unset for timings. Actual commands, GPU UUIDs,
weight/runtime hashes and resource records are retained with each experiment.

The flag accepts only exact 0/1 and is mutually exclusive with the research
`STRATA_EP_L0` hook. Unsupported topology, insufficient capacity and helper
initialization failures are rejected. EP rejects peer-cache/peer-prefill modes,
`--remote-expert-opt`, `--batch` and `--slots`. Their interaction is unvalidated.
Seven helpers hold 9,216 verified expert replicas, covering 72 helper–layer
pairs. Capacity must be checked for each actual launch.

Initialization publishes ownership only after success. Partial submissions keep
pending ownership until drained; failed synchronization cannot publish stale
result rows. Real sticky device failure still requires process termination and
a healthy device before restart. No driver reset or device-loss injection is
part of this project.

## Historical performance

On upstream `99f3dbd`, engine source `889028a`, the previous final four-boot
OFF/ON/ON/OFF matrix measured 60.08 → 62.62 decode tok/s (+4.22%), with client
elapsed reduced 3.06%. This includes the EP gain; it must not be multiplied into
another already-optimized result. The 36 performance requests used Chinese,
English and code at 1K/4K/16K, 256 output tokens and zero prompt reuse. QA was
4/5 with the known arithmetic error. Text/draft equality did not prove full
internal-state equivalence. See [historical matrix](AGC8F_EP24_FINAL_MATRIX.json).
Earlier limited screens remain in their separately attributed JSON files.

The 48-layer expansion is not promoted: its incremental short-screen gain over
EP24 was 1.78%, with mirrored aggregates 0.60%/2.99% and a negative long-input
pair. It failed its preregistered screen. No EP48 code is added here.

## Upstream maintenance

Preserve measured versions before rebasing this owned feature branch. Record
the exact upstream SHA, use independent builds and repeat device acceptance;
never attach old timing numbers to a new binary. Use force-with-lease only for
this feature branch. Do not rewrite the upstream mirror `main` or validated
`agc8f` branch. Keep fixed weights/precision, serialized GPU experiments and
identity-checked cleanup. No PD separation or system PCIe/driver changes.
