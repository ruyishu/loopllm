<div align="center">

<h1>loopllm</h1>

<h3>Looping MiniMind's 8 layers four times — a strict A/B on Looped Transformers</h3>

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776ab)](requirements.txt)
[![Base](https://img.shields.io/badge/base-MiniMind-8a2be2)](https://github.com/jingyaogong/minimind)
[![PRs](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/ruyishu/loopllm/pulls)

**Same architecture, same parameter count, same data, same hyper-parameters — the only variable is how many times the layer stack is executed.**

中文版见 [README.md](README.md)

</div>

---

## 📌 What this is

"Looped / recurrent-depth Transformers" are one of the defining architecture bets of 2025–2026: OpenAI's rumoured *Astra* recurrent depth, ByteDance's **Ouro-1.4B**, **LoopFormer** (ICLR 2026), **Fully Looped Transformer**.

Most published work compares under an *iso-parameter / iso-FLOP* budget by **cutting depth and adding loops** ("24 layers → 6 layers, loop 4×"). This repository asks a simpler question:

> **Keep the depth fixed. Execute the same stack a few more times. Does the extra compute buy performance?**

We take [MiniMind](https://github.com/jingyaogong/minimind) — a from-scratch 64M teaching repo covering pretrain → SFT → DPO → RLAIF → Agentic RL → eval — add **~30 lines** for `n_loops`, and run a strict A/B:

| | T=1 (vanilla MiniMind) | T=4 (loopllm) |
|---|---|---|
| Architecture | 8 layers / d768 / GQA 8Q-4KV | **identical** |
| Parameters | 63.91M | **63.91M** (weight-shared, 0 new params) |
| Data / hyper-params / hardware | mini or full tier, upstream defaults | **identical** (full-tier data md5-verified) |

---

## 🎯 Results at a glance

| Dimension | T=1 | T=4 | Change |
|---|---|---|---|
| Final pretrain loss (mini tier, same 39,695 steps) | 1.6496 | **1.5853** | **−3.9%, lower at every single step** |
| Final pretrain loss (**full tier**, 264,651 steps) | 1.6433 | **1.5898** | **−3.3%** |
| Final SFT loss (**full tier**, 319,340 steps) | 1.4955 | **1.3895** | **−7.1%** |
| 20-question math tool-use (mini tier) | 6/20 (30%) | **10/20 (50%)** | **+20 points, stable** |
| 20-question math tool-use (**full tier**) | 7/20 (35%) | **13/20 (65%)** | **+30 points — all four post-training stages land on 13/20** |
| vs. official upstream weights | 7/20 (35%) | **13/20 (65%)** | **the full tier now clearly beats the official full-data weights (9/20)** |
| Cost | — | **3.5–3.9× time per step** | params unchanged, **no VRAM saving** |

<div align="center">

![Downstream gain from looping at the same depth: +30 points on the full tier, +20 on the mini tier](figures/fig8_four_arm.png)

</div>

---

## 🔧 What changed: one switch, ~30 lines

All of it lives in `model/model_minimind.py` (`MiniMindModel.forward`):

1. **Per-round residual gate (LayerScale style)**, initialised to `1/√T` so the T-round residual sum does not grow with T.
2. **Input re-injection**: from round `r > 0`, the first-round embedding is added in again, preventing representation drift across rounds.
3. **KV cache disabled when `n_loops > 1`** — a single cache slot cannot represent several passes. This is also why looping **does not save VRAM**.
4. `--n_loops` threaded through every trainer and evaluation script (**default 1 = upstream behaviour, bit-for-bit**).

---

## 📊 Experiments

### 1. Pretraining loss — same data, same steps, same parameter count

| Step | T=1 | T=4 | T=4 − T=1 |
|---|---|---|---|
| 1,000 | 2.1109 | 2.0785 | −0.032 |
| 10,000 | 1.8531 | 1.8193 | −0.034 |
| 30,000 | 1.7276 | 1.6910 | −0.037 |
| **39,695** | **1.6496** | **1.5853** | **−0.064 (−3.9%)** |

<div align="center">

![Pretraining loss comparison](figures/fig2_loss.png)
![Step-wise loss difference — negative throughout](figures/fig3_diff.png)

</div>

### 2. Downstream: 20-question math tool-use benchmark (self-built, auto-graded)

| Checkpoint | T=1 · mini data | **T=4 · mini data** | T=1 · full data | **T=4 · full data** |
|---|---|---|---|---|
| pretrain | 0/20 | 0/20 | 0/20 | 0/20 |
| full_sft | 6/20 (30%) | **10/20 (50%)** | 7/20 (35%) | **13/20 (65%)** |
| dpo | — | 10/20 (50%) | 7/20 (35%) | **13/20 (65%)** |
| grpo | 6/20 (30%) | 10/20 (50%) | 7/20 (35%) | **13/20 (65%)** |
| agent | 6/20 (30%) | 10/20 (50%) | 6/20 (30%) | **13/20 (65%)** |

(`—` = that checkpoint was not evaluated; under the `math_plus` tool set the trend is identical with lower absolutes — full tier goes 6/6/7/8 → **10/10/10/11**.)

The gap comes from **expression correctness** (full tier 8/20 → 14/20, mini tier 7/20 → 11/20) while tool-calling rate is 20/20 on both sides — the looped model learns to **build the arithmetic expression correctly**, not merely to call the tool more often.

<div align="center">

![Downstream gain from looping at the same depth (mini tier)](figures/fig1_downstream.png)

</div>

### 3. Versus the official upstream weights (same questions, same protocol, `n_loops=1`)

| Weights | Correct | Called tool | Expression correct |
|---|---|---|---|
| official `agent` | 9/20 (45%) | 20/20 | 11/20 (55%) |
| official `full_sft` | 9/20 (45%) | 17/20 | 10/20 (50%) |
| this repo, T=1 full data `full_sft` | 7/20 (35%) | 18/20 | 8/20 (40%) |
| **this repo, T=4 mini data `full_sft`** | **10/20 (50%)** | 20/20 | **11/20 (55%)** |
| **this repo, T=4 full data `full_sft`** | **13/20 (65%)** | 20/20 | **14/20 (70%)** |
| **this repo, T=4 full data `agent`** | **13/20 (65%)** | 20/20 | **14/20 (70%)** |

- The **official weights score 45%** on this benchmark ⇒ that is the 64M reference point; **the T=4 full tier pushes it to 65%**
- Knowledge benchmarks line up item-by-item with upstream (GSM8K: 4/300 both) and both sides sit on the floor there ⇒ this scale cannot measure knowledge; only tool-use style tasks separate checkpoints
- **T=4 matches official full-data weights with 1/8 of the data, and clearly beats them with the full data**

### 4. Cost

| Metric | T=1 | T=4 | Ratio |
|---|---|---|---|
| Steps/sec (full-tier SFT, seq 768 / batch 16) | 8.08 | 2.29 | 3.5× |
| Throughput (stable training) | 99,262 tok/s | 28,092 tok/s | 3.5× |
| Effective MFU (counting 4 passes) | 32.0% | ≈36% | comparable |

### 5. Side finding: **the size of the tool set is the real bottleneck at 64M**

| Tool set | Calls the calculator | Answer accuracy |
|---|---|---|
| 8 tools (all upstream tools) | 0 / 3 | 0% |
| 3 tools | 16–17 / 20 | 10–15% |
| 1 tool (calculator only) | 20 / 20 | 30% |

> Any tool-use number **must** state the tool-set size, otherwise it is not comparable. This repo reports `math_only` (calculator only) by default; `--tool_mode math_only|math_plus|all`.

<div align="center">

![Tool-set size is a hard bottleneck at 64M](figures/fig5_toolset.png)

</div>

---

## 🚀 Quick start

```bash
git clone https://github.com/ruyishu/loopllm.git && cd loopllm
pip install -r requirements.txt          # torch 2.6+ recommended; 48G single card for the full tier

# Pretraining: T=1 (upstream behaviour) vs T=4 (loop the stack 4×)
python trainer/train_pretrain.py --n_loops 1 --data_path ../dataset/pretrain_t2t_mini.jsonl
python trainer/train_pretrain.py --n_loops 4 --data_path ../dataset/pretrain_t2t_mini.jsonl

# Post-training: --n_loops must match the pretrained checkpoint
python trainer/train_full_sft.py --n_loops 4 --from_weight t4_pretrain
python trainer/train_dpo.py      --n_loops 4 --from_weight t4_sft
python trainer/train_grpo.py     --n_loops 4 --from_weight t4_dpo --loss_type cispo
python trainer/train_agent.py    --n_loops 4 --from_weight t4_grpo

# Evaluation: 20-question math tool-use, auto-graded (questions embedded in the script)
python scripts/eval_math_toolcall.py --n_loops 4 --weight t4_sft --tool_mode math_only

# Regenerate all report figures
python scripts/make_report_figs.py
```

Add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for long runs — without it you will hit random OOMs.

Datasets follow MiniMind's format; see [`dataset/dataset.md`](dataset/dataset.md). **T=1 and T=4 must use the identical data** — that is the precondition of the experiment.

---

## 🩹 Engineering notes (half of a long reproduction is spent here)

| Incident | Symptom | Root cause | Fix |
|---|---|---|---|
| **CRLF freeze** | Chain process alive but no progress for 11h | Script written with `\r\n`; `sleep 60\r` looped forever | Enforce LF + verify `CR=0` after deploy; **watch stage markers, not process liveness** |
| **RL OOM with T=4** | RLAIF/AGENT exit immediately | 4 passes alone exceed 44.5G; small config plus pretraining on the same card OOMs again | `batch 1 × num_generations 2` + `expandable_segments`; RL must run on an idle GPU |
| **Misleading tok/s** | tok/s drops 21,000 → 288 | Checkpoint saving inside the same log window | Judge by **steps/sec**; tok/s trend only |
| **Eval needs stdin** | `EOFError` | `input()` inside `eval_toolcall.py` | Pipe `echo 0 \|` in the chain |
| **Missing reward-model shard** | RL fails in seconds | Interrupted download left a partial file | Re-download + md5 verify |
| **"Duplicate training" false alarm** | 9 `train_full_sft.py` processes | Main process + 8 DataLoader workers share the command line | Check `ps -o pid,ppid` |
| **Reward function kills the whole run** | Agentic RL dies with `AttributeError` at step 2,412 | Argument validation only JSON-parses `arguments` when it is a **string**; a model emitting a **float** reaches `CHECK_ARGS[...]` and `.get()` raises | Coerce any non-dict to `{}` (scored as an invalid call) instead of letting it terminate training; same patch on **both arms**, backup taken first |
| **"Weight file exists = success"** | The rescue script treated crashed/truncated rounds as finished and skipped the retry tier | The trainer saves every 10 steps, so a crashed run still leaves a weight file behind | Judge success from the **log** (Traceback? exit code 0/124), never from file existence |

---

## ⚠️ Limitations (read together with the results)

- **Scale**: everything is measured at **64M parameters**; extrapolation to larger models/longer training is untested
- **Evaluation**: a single 20-question task, extremely sensitive to tool-set size; n=20 has limited resolution
- **RL comparison is weak and not hyper-parameter matched**: ① T=4 cannot finish an epoch on one card (RLAIF ≈3 days, AGENT ≈16 days), so the RL stages use an **iso-wall-clock** budget (a degraded control); ② matching T=1's RL config (`batch 2 / num_generations 4`) **OOMs immediately** (44.52 GiB with 234 MiB free), so the T=4 RL stages run `batch 1 / num_generations 2` — i.e. **equal data coverage, unequal hyper-parameters**. SFT / DPO are unaffected and remain clean equal-data equal-step controls
- **No T sweep**: only T=1 vs T=4, so "what is the optimal T" is unanswered
- **No iso-compute control**: "what if T=1 got the extra 3.5× compute" is deliberately out of scope (that is the next step)

---

## 🗺️ Roadmap

- [x] Full-tier T1 vs T4 reconciliation — all five checkpoints are now aligned (see the four-arm table); what remains is finishing the T=4 RL budget from its checkpoint
- [ ] Iso-compute curve (same wall-clock budget)
- [ ] T sweep (T=2 / 4 / 8) and the optimal T
- [ ] Scale extrapolation (104M / 198M tiers)
- [ ] Inference-side KV reuse for looped decoding

---

## 🙏 Credits & license

- A derivative work of **[MiniMind](https://github.com/jingyaogong/minimind)** (Apache-2.0). Data format, full training pipeline, model skeleton and evaluation assets come from upstream; our only architectural change is looped execution (see [`NOTICE`](NOTICE)). Upstream features (MoE / LoRA / distillation) remain intact and identical at T=1.
- Related work: Ouro-1.4B (arXiv:2510.25741), LoopFormer (arXiv:2602.11451), Fully Looped Transformer (arXiv:2605.18797), stability of looped transformers (arXiv:2604.15259).
- Licensed under **Apache-2.0**, as upstream.

<div align="center">

If this reproduction is useful to you, a ⭐ is appreciated — and tell us in the Issues which T or which base model you would like to see next.

</div>
