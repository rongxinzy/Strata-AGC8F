# AGC8F boundary and expert-plan diagnostics

`STRATA_GDN45_BOUNDARY_PROBE=1` samples the first full prefill chunk on the stage
owning layers 44/45. Its 14 markers cover layer-44 final residual, layer-45 initial
GDN state, mixed activations, projections, recurrence inputs and outputs.
`STRATA_GDN45_BOUNDARY_RAW=recurrence` retains five FP32 arrays; other markers
retain full-array digests and finite-value statistics. This uses checked owner
stream synchronization and host reads. All diagnostic timings are excluded.

`STRATA_VERIFY_PLAN_TRACE=/absolute/directory` records the CPU-produced expert
plan before publication. It separates resident and PCIe shares and retains group
size histograms by device, stage, layer, quantization pair and verify window.
It does not read the unused fourth producer count. Device-generated plans are
explicitly excluded. Both features are disabled by default.

On agc8f, frozen source `ba71f3ba029c7c8447b05e3a054be4305f9af3f1` built for
sm89 and passed 25 host CTests. Two independent eight-GPU boots completed eight
requests using IQ3_S, the original MTP and FP16 KV, with MTP batching off.
The initial GDN45 state was exactly all-zero bits; all 14 boundary digests agreed
between boots. The five retained arrays were byte-identical. The callback probe
also matched within both boots and across all seven chunks. All 136 retained
boundary/callback raw files matched Strata's own hash algorithm. This is bounded
non-reproduction under instrumentation, not a repair or proof of race-freedom.
The earlier first-boot difference remains unresolved. Boundary capture alone
added approximately 1.2 and 1.8 seconds of instrumentation per boot.

Each boot captured 4,752 plans, 384 histogram rows, all eight devices, 48 layers
and seven quantization pairs, with zero trace errors/drops. The resident share
contained 92,210 groups and 120,480 routed entries. Single-entry groups were
77.05% of groups but 58.97% of entries. After removing intrinsically singleton
T=1 windows, they were 72.12% of groups and 52.54% of entries. All these runs used
G=1: T>=2 still permits multi-entry expert groups. A host-known T=1 specialization
is feasible; arbitrary dynamic singleton groups need a separate device strategy.
These short diagnostic requests do not establish a representative workload mix
or any speed improvement.

Raw requests/outputs, closure hashes, cleanup receipts and independent
recomputation remain in the research workspace, iterations 009/011/013. Both
boots exited and all GPU memory/ports returned idle. Boot one needed owned-group
SIGKILL after SIGTERM; the controller comparison helper failed and was replaced
by local raw recomputation. Its deadline watchdog was not exercised. The shared
model's earlier arithmetic sanity failure remains; this is no quality ranking.
