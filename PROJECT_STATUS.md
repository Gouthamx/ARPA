# ARPA — Project Status

**Last updated:** 21 August 2026
**Overall completion: ~85%**

---

## Results

Last full 10-paper run: 20 August, NVIDIA backend,
`nemotron-3-super-120b-a12b` for both extraction and code generation.

| Stage | Result |
|---|---|
| PDF → text | 10/10 |
| Methodology extraction | 10/10 complete |
| Dataset resolution | 10/10 |
| Code generation | 10/10 |
| Code compiles | **10/10** |
| Code actually runs | **10/10** |
| Model learns on real data | **2/2 of the papers that can be trained** |

```
easy2  EMNIST     54.4%  vs  2.1% chance   (paper reports 78.0%)
easy3  Kuzushiji  73.2%  vs 10.0% chance   (paper reports 98.8%)
```

A 200-step run is not a reproduction of a paper trained for tens of epochs.
The gap is reported as context, not as a verdict.

> **Read this before quoting the table above.** Those figures come from the
> 20 August run. Three changes landed after it (reasoning suppression, a higher
> token ceiling, a bounded component list) and only VGG has been re-run since.
> The changes are expected to help, but "10/10 running" has not been
> re-measured against them. A fresh full run is the first thing to do.

---

## Architecture fidelity

This is the axis that took longest and still varies most by paper.

| Paper | model_name | components | Generated |
|---|---|---|---|
| medium1 ResNet | ResNet | 8 | `BasicBlock` + `ResNet`, 11.69M params (= ResNet-18 exactly) |
| hard3 DeiT | DeiT-B | 7 | `DeiTBlock` + `DeiT` |
| hard2 Bilinear CNN | B-CNN | 4 | — |
| medium2 VGG | VGG-D | **0** | `VGG_D`, but 2 convolutions where VGG-16 has 13 |
| easy3 Kuzushiji | PreActResNet-18 | 0 | — |
| medium4 MobileNetV2 | MobileNetV2 | 0 | — |
| easy1, easy2 | none | 0 | dataset papers, no single architecture |
| medium3, hard1 | none | 0 | architecture pass failed on that run |

Three papers extract real layer structure. The rest get the model's identity
but no components, so CodeGen has the name and must invent the layers.

---

## The defects that mattered

Each was invisible in the pass/fail numbers, and each needed different evidence
to find.

**1. A frequency penalty was corrupting every generated file.**
`CODEGEN_FREQUENCY_PENALTY = 0.4` suppresses repeated tokens, and Python
repeats commas, `=` and indentation more than anything else. It produced
arguments mangled into identifiers
(`nn.Conv2d(in_channels=1_out_channels_32_kernel_size_3)`). Measured 1/4 vs
4/4 compiling with it off; end-to-end 1/10 → 10/10.

**2. The PDF pipeline discarded three quarters of every paper.**
`max_chars=16000` overrode the extractor's own 32000 default, cutting a 12-page
paper from ~59k characters to ~14k and taking the architecture tables with it.

**3. The codegen prompt denied the architecture existed.**
It opened with "No specific architecture mentioned in paper." while listing
seven extracted components underneath, and never sent `model_name` or
`architecture_family` at all.

**4. The extraction model was too small for the larger context.**
After fix 2, `llama-3.1-8b` began dropping fields — one paper lost
`model_name`, another lost the channel from `input_shape` (`[1,28,28]` →
`[28,28]`), which crashed the generated code. Same code, only the model
changed: smoke 6/10 on 8b, 10/10 on nemotron-120b.

**5. The reasoning model narrated into its own token budget.**
On JSON calls, nemotron opened with "We are given a paper excerpt. We need to
extract..." and ran out of room before finishing the object. NVIDIA's
`detailed thinking off` directive took the same request from >16384 tokens and
truncated to 405 tokens of clean JSON.

None of these were model capability limits. Handed a proper spec, the same
model generates ResNet-50 with 25.56M parameters, matching the reference
exactly, first try.

---

## What is left

### 1. Re-measure  *(~1-2 h, mostly waiting)*
Three changes have landed since the last full run and only VGG has been
re-tested. Run all ten and confirm 10/10 still holds.

```bash
python verify_codegen_agent.py --backend nvidia --fresh --train
```

### 2. VGG extracts no components  *(prompt iteration)*
Its architecture pass used to fail outright; it now completes and identifies
VGG-D correctly, which is more precise than the family-level "ResNet" other
papers get. But `components` is 0, so CodeGen generates 2 convolutions where
VGG-16 has 13.

The instruction bounding the component list traded one failure for another:
output too long to finish before, too little now. It needs a bound that still
*requires* the grouped stages -- prompt work, not another budget increase.
Note that this bound applies to every paper and has only been tested on VGG;
it may have reduced components elsewhere too.

### 3. Most papers get identity but no layers  *(the real remaining gap)*
Only 3 of 10 extract real component structure. The rest hand CodeGen a name
and let it invent the architecture. This is the difference between
"paper-informed code generation" and reproduction.

### 4. Extraction is non-deterministic  *(open-ended)*
Component counts and dataset names shift between runs. One run read DenseNet's
dataset as CIFAR-10, the next as ImageNet -- both defensible, but the second
costs a trainable paper.

### 5. Groq client parity  *(~20 min)*
`groq_client.py` still has the `finish_reason` blindness fixed in
`nvidia_client.py`: a truncated response is reported as malformed JSON.

---

## What cannot be finished

**Full reproduction of all 10 papers is not achievable on this hardware.**

- 6 papers train on ImageNet-class datasets: ~150GB, gated registration, days
  of GPU time. They cannot run unattended.
- 1 paper compares sklearn estimators and has no neural model.
- 1 paper is generative, where accuracy is the wrong instrument.

The honest ceiling: **reproduces the feasible subset, generates runnable code
for the rest.**

---

## Model choice is load-bearing

Extraction reads ~28k characters per pass and a small model drops fields at
that size. The working configuration is recorded in `.env.example`; it is not
optional, because the failure is silent -- the pipeline reports success and the
generated code breaks later.

```
llama-3.1-8b                    smoke 6/10
nemotron-3-super-120b-a12b      smoke 10/10
```

If you edit the extraction prompts, note they go through `str.format`: a
literal `{` in an example will raise `KeyError` on every paper. That cost a
full debugging cycle.

---

## Reading the numbers honestly

- **"Runs" is not "reproduces."** 10/10 means the code executes and trains, not
  that it matches the paper's reported accuracy.
- **An earlier 10/10 was misleading.** Before smoke execution existed, the
  benchmark reported 10/10 while shipping a `train.py` that imported a function
  `model.py` never defined.
- **Results move between runs.** Accuracy shifts a few points and which papers
  are trainable can change with how a dataset name is extracted. The dependable
  claim is "clearly better than chance", not the decimals.
