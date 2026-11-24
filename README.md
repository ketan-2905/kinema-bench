# KinemaBench

Region-wise LoRA rank allocation and low-precision evaluation for multimodal
manipulation policies.

Fine-tuning a vision-language policy onto a new robot embodiment usually means
spreading a uniform LoRA rank across every adapted module and hoping. This
repository is built around three things that approach leaves out.

---

### 1. Rank placement, measured at a fixed trainable-parameter count

A manipulation policy has four functionally distinct regions: vision encoder,
multimodal projector, language backbone, action head. There is no reason to
expect a new embodiment stresses them equally. `alloc.py` distributes rank
across those regions from sensitivity scores, capped at **the same number of
trainable parameters a uniform rank-32 adapter would use**.

That constraint is the experiment. Without it, a variable-rank configuration
that wins has only proved that more parameters win, which needed no ablation.
The allocator asserts it never exceeds that parameter count.

### 2. The small-update problem under bf16

bf16 carries 8 mantissa bits. An optimiser update smaller than roughly 2⁻⁸ of
a parameter's own magnitude rounds straight back to the original value. No
error is raised. The loss curve still looks plausible, because the larger
updates early in training do land. The adapter quietly stops learning, and
this bites hardest at low LoRA rank, which is exactly where a variable-rank
allocation puts some regions.

`precision.py` holds every trainable tensor in fp32 master weights while the
frozen backbone stays in bf16, and then **proves it**: `assert_no_silent_erasure`
applies a realistically small update to each trainable tensor and raises if the
value did not move.

### 3. How a policy fails, not just how often

`failure_modes.py` labels each failed rollout as one of:

| Label | Meaning | Detection |
| --- | --- | --- |
| `ZERO_ACTION_COLLAPSE` | the policy stopped committing; the arm barely moves | action RMS below a floor |
| `INSTRUCTION_SWAP_GROUNDING_FAILURE` | acted confidently on the wrong instruction | replay identical observations under a *different* task's instruction; if the trajectory barely changes, the language is not being read |
| `SPATIAL_MISS` | grounded the instruction, moved decisively, still missed | residual, once the two above are ruled out |

The instruction-swap probe is the one worth the compute: it separates "the
model became imprecise" from "the model stopped listening", and those have
opposite fixes.

---

## Modules

| File | What it does |
| --- | --- |
| `regions.py` | maps a parameter name to one of the four functional regions |
| `alloc.py` | variable-rank allocation at a fixed parameter count, plus the uniform control arm |
| `precision.py` | fp32 master weights and the silent-erasure probe |
| `quantize.py` | INT8 and 4-bit NF4 round trips, middle-block depth ablation |
| `failure_modes.py` | the three-label attributor |
| `audit.py` | content-hashed, hash-chained result records |

### On quantisation off-GPU

Without bitsandbytes kernels, `quantize.py` performs a genuine quantise →
dequantise round trip through the real QLoRA NF4 codebook. The **accuracy**
degradation measured that way is real quantisation error. **Latency and memory
are not representative**, and those reports are stamped `kernels="simulated"`
so they cannot be mistaken for the real thing.

### On the audit trail

Each record's filename *is* the sha256 of its own content, it is written
read-only, and it carries the hash of the previous record. Editing a number
after the fact changes its filename; editing one in place fails `verify()`.

---

## Status

**Components only. No benchmark numbers have been produced.**

What exists is the measurement machinery: the allocator, the precision guard,
the quantisation round trips, the failure attributor and the audit store. What
does not yet exist is the training loop, the simulation rollout harness, and
therefore any result. Nothing in this repository reports a success rate,
a speedup or a comparison, because none has been run.

Next: wire the rollout harness to a manipulation suite, then run the uniform
and variable-rank arms at an equal parameter count across BF16, INT8 and NF4.

## References

- OpenVLA-OFT, parallel decoding and action chunking — arXiv:2502.19645
- QLoRA NF4 quantisation — arXiv:2305.14314
