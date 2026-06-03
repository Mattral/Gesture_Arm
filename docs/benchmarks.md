# Benchmarks

This repository does not include a canonical set of published benchmark numbers.
Instead, it provides runtime telemetry and MFU accounting so users can measure
performance on their own hardware and compare results transparently.

## What to measure

The runtime emits metrics for the following performance dimensions:

- `mfu`: model FLOP utilization computed by `pkg/utils/mfu.py`.
- `tokens_per_sec`: example system throughput in the training loop.
- `wall_clock_ms`: per-step wall-clock duration.
- `kernel` profile fields such as:
  - `sram_bytes_per_block`
  - `achieved_bw_gbps`
  - `tokens_per_expert_mean`
  - `tokens_per_expert_std`
  - `used_triton`
- `infra` fields such as:
  - `all_to_all_dispatch_ms`
  - `all_to_all_combine_ms`

These metrics are surfaced through the structured logger and the step JSON
output configured by `telemetry.json_path`.

## How to benchmark

1. Install dependencies and activate the environment.
2. Use a stable config for the target hardware.
3. Run the training loop with telemetry enabled:
   ```bash
   python train.py --config configs/default.yaml
   ```
4. Examine the JSON telemetry file from `telemetry.json_path`, or open the
   TensorBoard directory in `telemetry.tensorboard_dir`.

For lower-cost validation, use `configs/smoke.yaml` and the `--smoke` flag.
This exercises the same runtime paths with a tiny model.

## Useful benchmark targets

### Kernel-level performance

Measure the router and expert dispatch performance through:

- `kernel.achieved_bw_gbps`
- `kernel.sram_bytes_per_block`
- `kernel.used_triton`

These values are populated from the router profile and represent the lower-level
kernel characteristics.

### Collective and overlap

Look at the latency fields for EP communication:

- `infra.all_to_all_dispatch_ms`
- `infra.all_to_all_combine_ms`

The code aims to overlap these collectives using a dedicated CUDA stream.

### End-to-end throughput

Use `tokens_per_sec` and `wall_clock_ms` from the step records to compare
aggregate performance across different hardware and topology settings.

### MFU analysis

`pkg/utils/mfu.py` calculates MFU from the model's effective FLOPs and
hardware peak throughput. Its unit tests in `moe-engine/tests/test_mfu.py`
validate the expected behavior and scaling of the MFU formula.

## Notes on reproducibility

- Benchmarks are highly hardware-dependent.
- Use consistent config and environment variable settings across runs.
- Record the `world_size`, `device`, and `config` values alongside metric outputs.

## When to add benchmark results

If you capture sustained, reproducible performance data on real hardware, add
it to this document or a dedicated `benchmarks/` directory with:

- exact config file used
- hardware description
- command line
- telemetry summary
- any caveats on tuning or environment

Until then, this document remains a guide for how to measure and interpret
performance using the repository's existing telemetry.
