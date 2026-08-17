# ARPA — Project Status

**Last updated:** 17 August 2026
**Overall completion: ~75%**

---

## What that 75% means

The pipeline runs end to end and every stage is implemented and tested. What is
missing is not features — it is **evidence at scale**. The final stage has been
proven on one paper, not ten, and a third of the generated code still does not
execute.

A useful way to read the number:

| | |
|---|---|
| Infrastructure & pipeline | ~95% — all 7 stages built, 261 tests |
| Measured quality of output | ~55% — 10/10 compile, 7/10 run, 1 verified learning |
| Full reproduction (match paper accuracy) | ~10% — structurally blocked, see below |

---

## The pipeline as it stands

| # | Stage | State | Evidence |
|---|-------|-------|----------|
| 1 | PDF → text | Done | 10/10 papers |
| 2 | Methodology extraction | Done | 0 failures across 40 passes |
| 3 | Dataset resolution | Done | 10/10 papers |
| 4 | Code generation | Done | 10/10 papers |
| 5 | Syntax validation | Done | 10/10 compile |
| 6 | Smoke execution | Done | **7/10 actually run** |
| 7 | Training + metric comparison | Built, barely exercised | **1 paper verified** |

Best single result — the project's core claim, demonstrated:

```
easy2 (EMNIST)   acc 64.9%   paper 96.2%   gap +31.3%   learning
```

ARPA read the paper, generated a model, trained it on real data, and it
learned (64.9% against 2.1% chance).

---

## What is left, in priority order

### 1. Run stage 7 across all 10 papers  *(~1–2 h, mostly waiting)*
The training stage has produced exactly one real result. Until it runs over the
whole benchmark there is no aggregate claim, only an anecdote.

```bash
python verify_codegen_agent.py --backend nvidia --train --fresh
```

Expect roughly 3 trainable papers (EMNIST, Fashion-MNIST, Kuzushiji), 1 CIFAR
paper that is slow but feasible, and 6 ImageNet papers that will skip. Record
the numbers; that table is the project's headline result.

### 2. Fix the 3 papers whose code does not run  *(~2–3 h)*
From the last smoke run, all three are real defects in generated code:

| Paper | Defect |
|-------|--------|
| medium4 | assigns a submodule before calling `super().__init__()` |
| hard1 | forward pass fails on a stride/dimension mismatch |
| hard2 | `dataset_loader.py` has a syntax error on line 1 |

These are prompt/codegen problems, not infrastructure. Each fix needs a full
rerun (~50 min) to validate, and results vary between runs, so budget
generously.

### 3. Syntax-check `dataset_loader.py`  *(~15 min)*
`check_generated_code()` only compiles files produced by CodeGenAgent.
`dataset_loader.py` comes from DatasetAgent and is never checked — which is
exactly how hard2 shipped a file that does not parse. Cheap, self-contained,
no rerun required.

### 4. Stabilise extraction  *(open-ended)*
Component counts swing between runs (0 ↔ 10 for the same paper), and
`max_iterations` appears and disappears. Nothing is *failing*; the model simply
returns different amounts of detail each time. Would need either a stronger
extraction model, self-consistency across repeated passes, or accepting the
variance and reporting a range.

### 5. Port the client fixes to Groq  *(~20 min)*
`groq_client.py` still has the `finish_reason` blindness that was fixed in
`nvidia_client.py`: a truncated response is reported as malformed JSON. Only
matters if Groq is used as a backend again.

---

## What cannot be finished, and why

**Full reproduction of all 10 papers is not achievable** — not with more time,
not on this machine.

- 6 of the 10 papers train on **ImageNet**: ~150 GB, gated behind manual
  registration, days of GPU time. These can never run unattended.
- CIFAR-scale papers need hours on a GPU for a real schedule.
- Only the MNIST-family papers are trainable in minutes.

So the honest ceiling for this benchmark is:

> reproduces the feasible subset, generates runnable code for the rest

Anyone claiming full 10/10 reproduction on this hardware is describing
something that did not happen.

---

## Known limitations worth stating in a write-up

- **"Learns" is not "reproduces."** Stage 7 runs 150 steps; papers train for
  tens of epochs. The gap is reported as context, never as a verdict.
- **Accuracy is the wrong instrument for some papers.** VAEs and other
  generative models are skipped rather than scored — deliberately.
- **The benchmark measures the environment too.** Missing `timm` /
  `tensorflow_datasets` are reported as skipped, not failed, so the score
  reflects the code rather than the virtualenv.
- **10/10 "compiles" was misleading**, which is why smoke execution exists:
  a `train.py` importing a function `model.py` never defined compiled cleanly
  and passed.

---

## Suggested definition of done

For a project of this scope, "complete" is reasonably:

- [x] All 7 stages implemented and tested
- [x] Extraction failures eliminated (8 → 0)
- [x] Generated code compiles (10/10)
- [ ] Generated code runs (7/10 → 10/10)
- [ ] Training verified across every feasible paper (1 → ~4)
- [ ] Results documented with the numbers above

Three of six done, two partially. Hence ~75%.
