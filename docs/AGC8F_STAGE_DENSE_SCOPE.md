> Upstream `6f32ec0` includes early stage trimming through `--trim-stage-weights`
> or `STRATA_STAGE_TRIM`. When that path is active, `STRATA_STAGE_DENSE_SCOPE`
> does not reload already-scoped tables. With early trim off, the legacy late
> scope remains opt-in and uses upstream native range rules to retain PLE.
> Historical allocation/speed results below have not been repeated on this upstream.

# Stage-owned dense weights on AGC8F

`STRATA_STAGE_DENSE_SCOPE=1` is an optional memory-residency change. It is off unless the value is exactly `1`. Later layer-split stages upload only their owned dense layers. The main stage stays fully loaded; globals and unparseable names remain present. Canonical and native loaders retain their existing complementary formats. The full initial load still precedes split selection. Replacement finishes before any session, verifier, prefill or MTP reader binds the weight objects.

## Device validation

Client date 2026-10-04, agc8f-6: eight RTX 4060 Ti 16GB, Hygon AVX2 CPU, 125GiB RAM, shared PCIe4 x8 CPU uplink. Frozen source `75382b13c9e28cbcd53aa3770cf319b1101571aa`; sm89 Release executable SHA-256 `0427eb0374769ffeaf236923d0bd2847403a07f3eaa2f5aad5e38279bb28cb6e`. Source archive SHA-256 `fa61efcb8937a65a6af924c578a3d23eaccadcca5f9f406ec2a2e6fd6a0dd062`. One build and the same executable path were used for OFF/ON/ON/OFF; per-boot executable hashes were not taken. CUDA build and all 26 CTest cases passed.

The four boots made 60 requests: four warmups, 20 sanity requests and 36 held-out performance requests. Formal inputs were 1K/4K/16K across Chinese, English and code, each generating 256 tokens without prompt reuse. Fixed GSQ-RCO IQ3_S main, original MTP, FP16 KV, 2048-token prefill chunks and explicit six-layer stages were used. Performance runs had no numerical audit or profile instrumentation.

| Residency evidence | OFF | ON | Reduction |
| --- | ---: | ---: | ---: |
| Sum of per-card sampled maximum used memory | 95764 MiB | 73098 MiB | 22666 MiB / 22.14 GiB |
| Logged canonical + native weight allocation bytes, stages 1–7 | — | — | 22359766912 bytes / 20.82 GiB |

The first row sums each card's maximum from one-second samples. It is not a simultaneous machine peak or a measurement of transient startup peak. Both boots within each arm had the same sampled maxima. GPU0 was unchanged; GPU1–7 each decreased by about 3.2GiB, including GPU7 (3242MiB). The difference between sampled residency savings and logged weight bytes has not been attributed.

All four boots retained 3072 cache slots per stage, 100% decode cache hits and independent prompt buffers. No cache-loan/own-buffer selector changed. The coarse experts-loaded log counters were 114/115 seconds OFF and 120/125 seconds ON; reloading is not free, and startup transient allocation remains a limitation.

## Performance and correctness bounds

Across the nine formal labels, total engine prefill+decode time ON/OFF was `1.0048697` and client elapsed time ON/OFF was `0.9956867`. These small opposite changes are within boot variation; only two boots per arm were measured. No stable throughput gain or quality-controlled ranking is accepted.

A separate four-boot OFF/ON/ON/OFF capacity experiment used the same source and executable with 4096-token prefill chunks. Its 60 requests comprised four warmups, 20 sanity requests and 36 independent calibration requests; the calibration corpus did not replace the held-out results above. Both arms still selected independent buffers on all stages: buffer demand was 2.06GiB and the OFF last stage's 3.59GiB free memory narrowly exceeded demand plus the existing 1.5GiB reserve. The expected cache-loan branch change did not occur, so this experiment did not validate the proposed capacity-to-speed mechanism. No larger-chunk sweep followed.

The sum of per-card sampled maxima was 101954MiB OFF and 79288/79286MiB ON, again approximately 22.14GiB less; this remains a sampled residency sum, not a simultaneous or transient peak. Calibration client elapsed time ON/OFF was approximately `1.00333`, engine service time `0.99915`. No stable speed improvement is accepted. Boots 2 ON, 3 ON and 4 OFF matched all 15 output texts; boot 1 OFF differed on seven. All sanity answers retained the same known 4/5 result. Raw answer-validation failures represent the wrong arithmetic answer, independently of any short-answer EOS.

Boots 1 OFF, 2 ON and 3 ON matched all 15 output text/reasoning pairs. Boot 4 OFF differed on eight, including seven formal requests and the warmup. This OFF/OFF counterexample leaves startup-dependent numerical behavior unresolved; it does not prove the flag caused or removed divergence. All five sanity answers matched across boots, retaining the known arithmetic answer 4032 instead of 4014 (4/5). Generated token IDs were not exposed by the API, and no numerical state audit ran in these timing trials. Text/count checks do not establish bitwise GPU-state equivalence or comprehensive quality.

## Reproduction

The isolated run used this localhost server launch, with the ON flag stored in the config's engine environment:

```sh
LD_LIBRARY_PATH=/usr/local/cuda/lib64 PYTHONUNBUFFERED=1 taskset -c 0-15 \
  /root/strata-benchmark/venv/bin/python -m serve.server --engine strata \
  --config /root/strata-benchmark/results/2026-10-04-autoresearch-015/scope-2-on/config.json \
  --host 127.0.0.1 --port 18080
```

Engine env: `STRATA_VERIFY_DEVICE_PLAN=0`, `STRATA_PREFILL_PIPELINE=1`, `STRATA_MTP_BATCH_MULTI=0`, `STRATA_STAGE_DENSE_SCOPE=1`. Engine config: GPUs 0–7; layer cuts `6,12,18,24,30,36,42`; `--prefill 2048 --spec 4 --spec-min-p 0.5 --max-context 32768 --kv fp16 --pool-workers 15 --pool-affinity all --prompt-cache 0 --prompt-cache-every 0 --conversation-cache-mib 0 --suffix-draft 3 --ple-io ram --adapt-swaps 0 --no-spec-split`, original expert profile and automatic cache. Full paths, arguments, affinity, runtime closure hashes, weights SHA and raw records are retained in the result tree.

Evidence in the research workspace: `autoresearch/agc8f/iterations/015/dense-scope-device/results/2026-10-04-autoresearch-015/`. Independent review is in iteration018, with root corrections in `iterations/018/root-acceptance/acceptance.md`. The independent review's prose incorrectly paired boot4 with boot3; its raw diff matrix and root recomputation retain the eight differences. Execution prose that claimed GPU7 unchanged or startup peak measured is likewise superseded here.

Only saved owned PIDs were stopped. Final audit found all cards idle, localhost ports closed and the global experiment lock released. The watchdog did not fire; recovery is not claimed tested. The feature remains default-off and unmerged.
