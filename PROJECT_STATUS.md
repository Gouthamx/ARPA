# ARPA — Project Status

**Last updated:** 20 August 2026
**Overall completion: ~85%**

---

## Current results

Measured on the 10-paper benchmark, NVIDIA backend, `nemotron-3-super-120b-a12b`
for both extraction and code generation.

| Stage | Result |
|---|---|
| PDF → text | 10/10 |
| Methodology extraction | 10/10 complete (5 individual pass failures, degraded not fatal) |
| Dataset resolution | 10/10 |
| Code generation | 10/10 |
| Code compiles | **10/10** |
| Code actually runs | **10/10** |
| Model learns on real data | **2/2 of the papers that can be trained** |

Training results:

```
easy2  EMNIST     54.4%  vs  2.1% chance   (paper reports 78.0%)
easy3  Kuzushiji  73.2%  vs 10.0% chance   (paper reports 98.8%)
```

Both clear chance decisively. A 200-step run is not a reproduction of a paper
trained for tens of epochs -- the gap is reported as context, not as a verdict.

Architecture fidelity, the thing that took longest to get right:

| Paper | Extracted | Generated |
|---|---|---|
| ResNet | ResNet, 8 components | `BasicBlock` + `ResNet`, 11.69M params (= ResNet-18 exactly) |
| DeiT | DeiT-B, 7 components | `DeiTBlock` + `DeiT` |
| VGG | nothing, 0 components | generic CNN — **still unfixed** |

---

## The three defects that mattered

Worth recording, because each was invisible in the pass/fail numbers and each
took a different kind of evidence to find.

**1. A frequency penalty was corrupting every generated file.**
`CODEGEN_FREQUENCY_PENALTY = 0.4` suppresses repeated tokens -- and Python
repeats commas, `=` and indentation more than anything else. It produced
arguments mangled into identifiers
(`nn.Conv2d(in_channels=1_out_channels_32_kernel_size_3)`). Measured 1/4 vs
4/4 compiling with it off; end-to-end 1/10 -> 10/10.

**2. The PDF pipeline discarded three quarters of every paper.**
`max_chars=16000` overrode the extractor's own 32000 default, cutting a 12-page
paper from ~59k characters to ~14k and taking the architecture tables with it.
ResNet's Table 1 never reached extraction, so the architecture pass returned
three vague components and CodeGen filled the gap from the model's own priors.

**3. The codegen prompt denied the architecture existed.**
It opened with "No specific architecture mentioned in paper." whenever notes
were empty -- while listing seven extracted components directly underneath --
and never sent `model_name` or `architecture_family` at all. A run that had
correctly extracted "ResNet" told the generator the architecture was unknown.

None of these were model limitations. Handed a proper spec, the same model
generates ResNet-50 with 25.56M parameters, matching the reference exactly,
first try.

---

## What is left

### 1. VGG extracts nothing  *(~30-60 min)*
`medium2` returns 0 components and falls back to a generic CNN. Every other
paper now produces a real architecture, so this is a specific failure rather
than a general limit.

### 2. Exact-variant fidelity  *(open-ended)*
ResNet generates ResNet-18 (11.69M) where the paper headlines ResNet-50
(25.6M) -- right family, wrong size. DeiT already resolves the variant
(DeiT-B), so this looks reachable rather than fundamental.

### 3. Extraction is non-deterministic  *(open-ended)*
Component counts and dataset names shift between runs. One run read DenseNet's
dataset as CIFAR-10, the next as ImageNet -- both defensible, but the second
costs a trainable paper. Would need self-consistency across repeated passes,
or accepting the variance and reporting a range.

### 4. Groq client parity  *(~20 min)*
`groq_client.py` still has the `finish_reason` blindness that was fixed in
`nvidia_client.py`: a truncated response is reported as malformed JSON.

---

## What cannot be finished

**Full reproduction of all 10 papers is not achievable on this hardware.**

- 6 papers train on ImageNet-class datasets: ~150GB, gated registration, days
  of GPU time. They cannot run unattended.
- 1 paper compares sklearn estimators and has no neural model to train.
- 1 paper is generative, where accuracy is the wrong instrument.

So the honest ceiling is: **reproduces the feasible subset, generates runnable
code for the rest.** Anyone claiming 10/10 full reproduction on this machine is
describing something that did not happen.

---

## Model choice is load-bearing

Extraction reads ~28k characters per pass, and a small model drops fields at
that size. On `llama-3.1-8b` one paper lost `model_name` entirely and another
lost the channel from `input_shape` (`[1,28,28]` -> `[28,28]`), which crashed
the generated code. Same code, same prompts, only the model changed:

```
llama-3.1-8b                    smoke 6/10
nemotron-3-super-120b-a12b      smoke 10/10
```

The working configuration is recorded in `.env.example`. It is not optional --
the 8b model produces a pipeline that looks fine until you run the code.

---

## Reading the numbers honestly

- **"Runs" is not "reproduces."** 10/10 means the code executes and trains;
  it does not mean it matches the paper's reported accuracy.
- **An earlier 10/10 was misleading.** Before smoke execution existed, the
  benchmark reported 10/10 while shipping a `train.py` that imported a
  function `model.py` never defined. The metric could not tell parsing from
  working.
- **Results move between runs.** Accuracy shifts a few points, and which
  papers are trainable can change with how a dataset name is extracted. The
  dependable claim is "clearly better than chance", not the decimals.
