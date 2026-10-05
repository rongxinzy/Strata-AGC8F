# AGENTS.md

## AGC8F fork

This fork targets RongxinZY's agc8f (8 RTX 4060 Ti 16GB, Hygon AVX2 CPU,
125GiB RAM, shared PCIe4 x8 CPU uplink). See [docs/AGC8F.md](docs/AGC8F.md)
for source provenance, measured results and limitations. Keep upstream attribution
and the MIT license. `main` preserves upstream; `agc8f` holds device-validated changes.

- New optimized paths are opt-in until device validation supports enabling them.
- Keep measured IQ3_S main/MTP weights and FP16 KV fixed. Do not remove experts,
  PLE or change model/quantization to obtain speed numbers.
- Preserve raw performance and numerical diagnostics separately; profile/audit
  runs are excluded from formal timings. Host tests do not prove CUDA correctness.
- The five-item sanity set has a known arithmetic failure. Retain it and all
  text/state differences; do not present runtime diagnostics as quality rankings.
- Use isolated builds/results and owned PID identity checks; do not stop unrelated
  services or change drivers, system clock, power limits or PCIe ACS configuration.
- `STRATA_BUILD_AGC8F_TESTS=ON` enables standalone host regression targets without
  requiring the unpublished upstream full test tree.

Strata runs the Qwen3.8-Flash-Next mixture-of-experts model (and its Coder, Swift 1.5 and Unsloth variants) on a
normal PC: one NVIDIA or AMD graphics card plus system RAM, on Windows or Linux. It has a C++/CUDA/HIP engine
(`src/`, `include/`), a Python server with an OpenAI- and Anthropic-compatible API and a web app (`serve/`), and a
one-click installer (`setup.py`, started by `START-HERE.bat` / `setup.sh`).

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**: check the PC, pick the model by RAM, run setup non-interactively,
start and verify the server, and connect the user's apps. Never expose the server beyond `127.0.0.1` without
`--api-key`. As an alternative to shell commands, Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md))
offers the same steps as tools.

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- Setup's own tests run without a GPU or downloads: `python tools/test_setup_<name>.py` (for example
  `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.
